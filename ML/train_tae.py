#!/usr/bin/env python3
"""Train a UTM-localized bidirectional-LSTM trajectory autoencoder.

Run from ``mobilitynet-analysis-scripts``::

    python ML/train_tae.py --epochs 100 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from pyproj import Transformer
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from .tae import TrajectoryAutoencoder, reconstruction_loss
except ImportError:
    from tae import TrajectoryAutoencoder, reconstruction_loss


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "datasets" / "trajectories" / "trajectory_samples.jsonl"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "tae_training"


class TrajectoryDataset(Dataset):
    """Load each JSONL item as localized UTM easting/northing and elapsed seconds."""

    def __init__(self, data_path: Path, max_trajectories: int | None = None) -> None:
        self.trajectories = self._load(data_path, max_trajectories)
        if not self.trajectories:
            raise ValueError("No trajectories with points were found in %s" % data_path)

    @staticmethod
    def _load(data_path: Path, max_trajectories: int | None) -> list[Tensor]:
        trajectories = []
        transformers: dict[int, Transformer] = {}
        with data_path.open() as data_file:
            for line in data_file:
                entry = json.loads(line)
                points = entry["tile_points"]
                if not points:
                    continue
                epsg = int(entry["epsg"])
                transformer = transformers.setdefault(
                    epsg, Transformer.from_crs("EPSG:4326", "EPSG:%d" % epsg, always_xy=True))
                southwest_easting, southwest_northing = entry["bounds_utm_m"][:2]
                start_timestamp = points[0]["ts"]
                localized_points = []
                for point in points:
                    easting, northing = transformer.transform(point["longitude"], point["latitude"])
                    localized_points.append((
                        easting - southwest_easting,
                        northing - southwest_northing,
                        point["ts"] - start_timestamp,
                    ))
                trajectories.append(torch.tensor(localized_points, dtype=torch.float32))
                if max_trajectories is not None and len(trajectories) >= max_trajectories:
                    break
        return trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> Tensor:
        return self.trajectories[index]


class FeatureStandardizer:
    """Normalize UTM offsets and elapsed times using training trajectories only."""

    def __init__(self, mean: Tensor, standard_deviation: Tensor) -> None:
        self.mean = mean
        self.standard_deviation = standard_deviation.clamp_min(1e-6)

    @classmethod
    def fit(cls, trajectories: list[Tensor]) -> "FeatureStandardizer":
        points = torch.cat(trajectories)
        return cls(points.mean(dim=0), points.std(dim=0, unbiased=False))

    def transform(self, trajectory: Tensor) -> Tensor:
        return (trajectory - self.mean) / self.standard_deviation

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean, "standard_deviation": self.standard_deviation}


class PreloadedTrajectoryDataset(Dataset):
    """Hold normalized trajectories and lengths on one device."""

    def __init__(self, trajectories: list[Tensor], device: torch.device) -> None:
        self.samples = [
            (
                trajectory.to(device),
                torch.tensor(len(trajectory), device=device, dtype=torch.long),
            )
            for trajectory in trajectories
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.samples[index]


def collate_trajectories(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
    """Sort and pad samples that are already resident on the accelerator."""
    batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    trajectories, lengths = zip(*batch)
    trajectories = torch.nn.utils.rnn.pad_sequence(trajectories, batch_first=True)
    lengths = torch.stack(lengths)
    point_indices = torch.arange(trajectories.shape[1], device=trajectories.device).unsqueeze(0)
    valid_points = (point_indices < lengths.unsqueeze(1)).unsqueeze(-1).to(trajectories.dtype)
    return trajectories, lengths, valid_points


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similar-length trajectories without exceeding a dense-padding budget."""

    def __init__(
        self,
        indices: list[int],
        lengths: list[int],
        batch_size: int,
        max_padded_points: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.indices = sorted(indices, key=lambda index: lengths[index], reverse=True)
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_padded_points = max_padded_points
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batches = self._make_batches()

    def _make_batches(self) -> list[list[int]]:
        batches = []
        batch = []
        for index in self.indices:
            padded_points = self.lengths[batch[0]] * (len(batch) + 1) if batch else self.lengths[index]
            if batch and (len(batch) == self.batch_size or padded_points > self.max_padded_points):
                batches.append(batch)
                batch = []
            batch.append(index)
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self.batches.copy()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
            self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self.batches)


def accelerator_device(requested_device: str) -> torch.device:
    """Select CUDA, then Apple MPS, then CPU unless explicitly requested."""
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if requested_device != "auto":
        if not available[requested_device]:
            raise RuntimeError("Requested %s accelerator is unavailable" % requested_device)
        return torch.device(requested_device)
    return torch.device(next(name for name in ("cuda", "mps", "cpu") if available[name]))


def mean_loss(model: TrajectoryAutoencoder, loader: DataLoader) -> float:
    """Evaluate the padding-aware reconstruction MSE."""
    model.eval()
    total_squared_error = None
    total_values = None
    with torch.no_grad():
        for trajectories, lengths, valid_points in loader:
            reconstruction, _ = model(trajectories, lengths)
            valid_element_count = valid_points.sum() * trajectories.shape[-1]
            squared_error = reconstruction_loss(reconstruction, trajectories, valid_points) * valid_element_count
            total_squared_error = squared_error if total_squared_error is None else total_squared_error + squared_error
            total_values = valid_element_count if total_values is None else total_values + valid_element_count
    return (total_squared_error / total_values).item()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-padded-points", type=int, default=8192,
                        help="Maximum batch_size * padded_sequence_length (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0 or args.max_padded_points <= 0:
        raise ValueError("--epochs, --batch-size, --hidden-dim, and --max-padded-points must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between zero and one")
    if args.max_trajectories is not None and args.max_trajectories <= 0:
        raise ValueError("--max-trajectories must be positive when provided")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw_dataset = TrajectoryDataset(args.data_path, args.max_trajectories)
    if len(raw_dataset) < 2:
        raise ValueError("Need at least two trajectories for training and validation")
    indices = list(range(len(raw_dataset)))
    random.Random(args.seed).shuffle(indices)
    validation_count = min(max(1, round(len(indices) * args.validation_fraction)), len(indices) - 1)
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]
    standardizer = FeatureStandardizer.fit([raw_dataset[index] for index in training_indices])
    device = accelerator_device(args.device)
    dataset = PreloadedTrajectoryDataset(
        [standardizer.transform(trajectory) for trajectory in raw_dataset.trajectories], device)
    if args.workers != 0 and device.type != "cpu":
        raise ValueError("--workers must be zero when trajectories are preloaded on an accelerator")
    trajectory_lengths = [len(trajectory) for trajectory in raw_dataset.trajectories]
    train_loader = DataLoader(
        dataset,
        batch_sampler=LengthBucketBatchSampler(
            training_indices, trajectory_lengths, args.batch_size,
            args.max_padded_points, shuffle=True, seed=args.seed),
        num_workers=args.workers, collate_fn=collate_trajectories)
    validation_loader = DataLoader(
        dataset,
        batch_sampler=LengthBucketBatchSampler(
            validation_indices, trajectory_lengths, args.batch_size,
            args.max_padded_points, shuffle=False, seed=args.seed),
        num_workers=args.workers, collate_fn=collate_trajectories)
    model = TrajectoryAutoencoder(hidden_dimensions=args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    print("Training %d trajectories (%d validation) on %s" % (
        len(dataset), len(validation_indices), device))

    with metrics_path.open("w") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for trajectories, lengths, valid_points in train_loader:
                optimizer.zero_grad(set_to_none=True)
                reconstruction, _ = model(trajectories, lengths)
                loss = reconstruction_loss(reconstruction, trajectories, valid_points)
                loss.backward()
                optimizer.step()
            training_loss = mean_loss(model, train_loader)
            validation_loss = mean_loss(model, validation_loader)
            metrics = {"epoch": epoch, "training_mse": training_loss, "validation_mse": validation_loss}
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            print("epoch=%d/%d train_mse=%.8f validation_mse=%.8f" % (
                epoch, args.epochs, training_loss, validation_loss))

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "feature_dimensions": model.feature_dimensions,
            "embedding_dimensions": model.embedding_dimensions,
            "hidden_dimensions": model.hidden_dimensions,
        },
        "feature_standardizer": standardizer.state_dict(),
        "features": ["easting_m_from_southwest", "northing_m_from_southwest", "seconds_from_start"],
    }, args.output_dir / "model.pt")


if __name__ == "__main__":
    main()