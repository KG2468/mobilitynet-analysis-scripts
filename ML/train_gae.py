#!/usr/bin/env python3
"""Train the encoder-only LPE graph autoencoder on road-network GraphML files.

Run from ``mobilitynet-analysis-scripts``::

    python ML/train_gae.py --epochs 50 --batch-size 16

The script prefers CUDA, then Apple MPS, and finally CPU. Use ``--device`` to
explicitly select one of ``cuda``, ``mps``, or ``cpu``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attr import has

import torch
from torch import optim
from torch.nn import functional as functional
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

try:
    from .gae import (
        DEFAULT_NODE_FEATURES,
        EncoderOnlyLPEGAE,
        FeatureStandardizer,
        RoadNetworkDataset,
        training_metadata,
    )
except ImportError:
    from gae import (
        DEFAULT_NODE_FEATURES,
        EncoderOnlyLPEGAE,
        FeatureStandardizer,
        RoadNetworkDataset,
        training_metadata,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "datasets" / "road_networks"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "gae_training"
DEFAULT_CACHE_PATH = REPOSITORY_ROOT / "datasets" / "gae_training" / "data.pt"
FOLD_COUNT = 5
CACHE_FORMAT_VERSION = 1


def accelerator_device(requested_device: str) -> torch.device:
    """Return the requested CUDA/MPS device, or select CUDA then MPS then CPU."""
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested_device != "auto":
        if not available[requested_device]:
            raise RuntimeError("Requested %s accelerator is unavailable" % requested_device)
        return torch.device(requested_device)
    for device_name in ("cuda", "mps", "cpu"):
        if available[device_name]:
            return torch.device(device_name)
    raise RuntimeError("No supported compute device is available")


def set_seed(seed: int) -> None:
    """Configure reproducible randomness where supported by the accelerator."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device, embedding_l2_weight: float) -> float:
    """Run one metadata-reconstruction epoch."""
    model.train()
    total_squared_error = 0.0
    total_values = 0
    for batch in loader:
        # Optimization: Asynchronous transfer if using pinned host memory
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, embeddings = model(batch.x, batch.lpe, batch.rwpe, batch.edge_index, batch.batch)
        reconstruction_loss = functional.mse_loss(reconstruction, batch.x)
        embedding_l2_loss = embeddings.square().mean()
        loss = reconstruction_loss + embedding_l2_weight * embedding_l2_loss
        loss.backward()
        optimizer.step()
        total_squared_error += reconstruction_loss.item() * batch.x.numel()
        total_values += batch.x.numel()
    return total_squared_error / total_values


def evaluate(model, loader, device) -> dict[str, float]:
    """Return aggregate and per-feature reconstruction MSE for the held-out fold."""
    model.eval()
    total_squared_error_by_feature = None
    total_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            reconstruction, _ = model(batch.x, batch.lpe, batch.rwpe, batch.edge_index, batch.batch)
            squared_error_by_feature = (reconstruction - batch.x).square().sum(dim=0)
            if total_squared_error_by_feature is None:
                total_squared_error_by_feature = squared_error_by_feature
            else:
                total_squared_error_by_feature += squared_error_by_feature
            total_nodes += batch.x.size(0)

    feature_mse = total_squared_error_by_feature / total_nodes
    metrics = {"test_mse": feature_mse.mean().item()}
    metrics.update({
        "test_mse_%s" % feature_name: mse.item()
        for feature_name, mse in zip(DEFAULT_NODE_FEATURES, feature_mse)
    })
    return metrics


def fold_indices(item_count: int, seed: int) -> list[list[int]]:
    """Return five deterministic held-out folds, each containing 20% of the data."""
    indices = list(range(item_count))
    random.Random(seed).shuffle(indices)
    return [indices[fold::FOLD_COUNT] for fold in range(FOLD_COUNT)]


def load_or_create_dataset(
    graph_paths: list[Path],
    cache_path: Path,
    lpe_dim: int,
) -> list:
    """Load a matching graph cache or rebuild it for the current feature schema."""
    cache_config = {
        "format_version": CACHE_FORMAT_VERSION,
        "feature_names": list(DEFAULT_NODE_FEATURES),
        "lpe_dim": lpe_dim,
        "graph_paths": [str(path) for path in graph_paths],
    }
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        if isinstance(cached, dict) and cached.get("config") == cache_config:
            return cached["dataset"]

    dataset = RoadNetworkDataset(graph_paths, lpe_dim=lpe_dim)
    raw_dataset = [dataset[index] for index in range(len(dataset))]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cache_config, "dataset": raw_dataset}, cache_path)
    return raw_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=4e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--embedding-l2-weight", type=float, default=1e-4,
                        help="L2 penalty coefficient for graph embeddings (default: %(default)s)")
    parser.add_argument("--scheduler", choices=("cosine", "restarts", "plateau"), default="cosine")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lpe-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--position-skip", type=bool, default=False,
                        help="Whether to add a positional encoding skip connection (default: %(default)s)")
    parser.add_argument("--GAT", type=bool, default=False,
                        help="Whether to use Graph Attention Network (GAT) layers (default: %(default)s)")
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lpe_dim < 0 or args.embedding_l2_weight < 0:
        raise ValueError("--epochs and --batch-size must be positive; --lpe-dim and --embedding-l2-weight cannot be negative")

    graph_paths = sorted(args.data_dir.glob("*.graphml"))
    if args.max_graphs is not None:
        if args.max_graphs <= 0:
            raise ValueError("--max-graphs must be positive when provided")
        graph_paths = graph_paths[: args.max_graphs]
    if len(graph_paths) < FOLD_COUNT:
        raise ValueError("Need at least %d GraphML files for %d-fold cross-validation" % (
            FOLD_COUNT, FOLD_COUNT))

    set_seed(args.seed)
    device = accelerator_device(args.device)
    
    raw_dataset = load_or_create_dataset(graph_paths, args.cache_path, args.lpe_dim)
    input_channels = raw_dataset[0].x.size(-1)
    if input_channels != len(DEFAULT_NODE_FEATURES):
        raise ValueError(
            "Dataset has %d features, but DEFAULT_NODE_FEATURES defines %d" % (
                input_channels, len(DEFAULT_NODE_FEATURES)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Training %d road networks with %d-fold cross-validation on %s" % (
        len(raw_dataset), FOLD_COUNT, device))
    
    use_vram_preload = device.type in ("cuda", "mps")

    for fold, test_indices in enumerate(fold_indices(len(raw_dataset), args.seed), start=1):
        test_index_set = set(test_indices)
        train_indices = [index for index in range(len(raw_dataset)) if index not in test_index_set]
        
        standardizer = FeatureStandardizer.fit(Subset(raw_dataset, train_indices))
        
        # 1. Transform dataset
        dataset = [standardizer.transform(raw_dataset[index]) for index in range(len(raw_dataset))]
        
        # 2. Optimization: Move dataset directly into Accelerator Memory (VRAM)
        # Eliminates CPU-to-GPU PCIe transfer during the training loop.
        if use_vram_preload:
            dataset = [data.to(device) for data in dataset]
            num_workers = 0  # GPU tensors cannot be unpickled across CPU worker processes
        else:
            num_workers = args.workers

        # 3. Optimization: Configure DataLoader for zero-copy / pinned memory
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(not use_vram_preload and device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        test_loader = DataLoader(
            Subset(dataset, test_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(not use_vram_preload and device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        model = EncoderOnlyLPEGAE(
            in_channels=input_channels,
            lpe_dim=args.lpe_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=256,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if args.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        elif args.scheduler == "restarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 10, T_mult=2, eta_min=1e-5)
        elif args.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5, min_lr=1e-5)
        fold_dir = args.output_dir / ("fold_%d" % fold)
        fold_dir.mkdir(exist_ok=True)

        with (fold_dir / "metrics.jsonl").open("w") as metrics_file:
            for epoch in range(1, args.epochs + 1):
                train_mse = train_epoch(model, train_loader, optimizer, device, args.embedding_l2_weight)
                test_metrics = evaluate(model, test_loader, device)
                metrics = {"fold": fold, "epoch": epoch, "train_mse": train_mse, **test_metrics}
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                if args.scheduler == "plateau":
                    scheduler.step(test_metrics["test_mse"])
                else:
                    scheduler.step()
                print("fold=%d epoch=%d/%d train_mse=%.8f test_mse=%.8f" % (
                    fold, epoch, args.epochs, train_mse, test_metrics["test_mse"]))
                print(" ".join(
                    "%s_mse=%.8f" % (feature_name, test_metrics["test_mse_%s" % feature_name])
                    for feature_name in DEFAULT_NODE_FEATURES
                ))

        # Checkpoint serialization remains on CPU for clean portability
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "feature_standardizer": standardizer.state_dict(),
            "model_config": {
                "in_channels": input_channels,
                "lpe_dim": args.lpe_dim,
                "hidden_dim": args.hidden_dim,
                "latent_dim": 256,
            },
            "metadata": training_metadata(DEFAULT_NODE_FEATURES, args.lpe_dim),
            "fold": fold,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }
        torch.save(checkpoint, fold_dir / "model.pt")


if __name__ == "__main__":
    main()
