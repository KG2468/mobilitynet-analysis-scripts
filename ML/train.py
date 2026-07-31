#!/usr/bin/env python3
"""Train the coordinate autoencoder with five-fold cross-validation on MPS.

Run from ``mobilitynet-analysis-scripts``::

    python ML/train.py --epochs 20 --batch-size 16
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from .cae import CoordinateAutoencoder, reconstruction_loss
except ImportError:
    from cae import CoordinateAutoencoder, reconstruction_loss


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "datasets" / "standard_map_renders"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "cae_training"
FOLD_COUNT = 5
SHEET_SAMPLE_COUNT = 10


class RenderDataset(Dataset):
    """Load RGB standard-map renders as normalized CHW tensors."""

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        pixels = torch.from_numpy(__import__("numpy").asarray(image).copy())
        return pixels.permute(2, 0, 1).float().div(255.0)


def mps_device():
    """Use Apple Metal acceleration; fail clearly if it is unavailable."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required but unavailable in this PyTorch installation")
    return torch.device("mps")


def fold_indices(item_count, seed):
    """Return five deterministic, non-overlapping validation index sets."""
    indices = list(range(item_count))
    random.Random(seed).shuffle(indices)
    return [indices[fold::FOLD_COUNT] for fold in range(FOLD_COUNT)]


def edge_temperature(epoch, epoch_count):
    """Linearly anneal soft-edge temperature from 5 to 15 inclusive."""
    if epoch_count == 1:
        return 15.0
    return 5.0 + 10.0 * epoch / (epoch_count - 1)


def mean_loss(model, loader, device, temperature, edge_weight, trajectory_color_weight):
    """Evaluate the combined reconstruction loss without retaining gradients."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            reconstructions, _ = model(images)
            total_loss += reconstruction_loss(
                reconstructions, images, edge_temperature=temperature,
                edge_weight=edge_weight,
                trajectory_color_weight=trajectory_color_weight).item() * len(images)
    return total_loss / len(loader.dataset)


def save_contact_sheet(model, dataset, validation_indices, device, output_path, seed):
    """Save ten random validation inputs and reconstructions side by side."""
    chosen_indices = random.Random(seed).sample(
        validation_indices, min(SHEET_SAMPLE_COUNT, len(validation_indices)))
    model.eval()
    rows = []
    with torch.no_grad():
        for index in chosen_indices:
            image = dataset[index].unsqueeze(0).to(device)
            reconstruction, _ = model(image)
            input_pixels = image[0].detach().cpu().permute(1, 2, 0).mul(255).clamp(0, 255)
            output_pixels = reconstruction[0].detach().cpu().permute(1, 2, 0).mul(255).clamp(0, 255)
            pair = torch.cat((input_pixels, output_pixels), dim=1).byte().numpy()
            rows.append(Image.fromarray(pair, mode="RGB"))

    sheet = Image.new("RGB", (256, len(rows) * 148), "white")
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(rows):
        top = row * 148
        sheet.paste(pair, (0, top + 20))
        draw.text((4, top + 3), "Input", fill="black")
        draw.text((132, top + 3), "Reconstruction", fill="black")
    sheet.save(output_path)


def train_fold(dataset, validation_indices, fold, args, device):
    """Train one fold and persist epoch metrics, weights, and contact sheets."""
    validation_set = set(validation_indices)
    training_indices = [index for index in range(len(dataset)) if index not in validation_set]
    generator = torch.Generator().manual_seed(args.seed + fold)
    training_loader = DataLoader(
        Subset(dataset, training_indices), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, generator=generator)
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    model = CoordinateAutoencoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    fold_dir = args.output_dir / ("fold_%d" % fold)
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = fold_dir / "metrics.jsonl"

    with metrics_path.open("w") as metrics_file:
        for epoch in range(args.epochs):
            temperature = edge_temperature(epoch, args.epochs)
            model.train()
            total_loss = 0.0
            for images in training_loader:
                images = images.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstructions, _ = model(images)
                loss = reconstruction_loss(
                    reconstructions, images, edge_temperature=temperature,
                    edge_weight=args.edge_weight,
                    trajectory_color_weight=args.trajectory_color_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(images)
            training_loss = total_loss / len(training_loader.dataset)
            validation_loss = mean_loss(
                model, validation_loader, device, temperature,
                args.edge_weight, args.trajectory_color_weight)
            sheet_path = fold_dir / ("epoch_%03d_contact_sheet.png" % (epoch + 1))
            save_contact_sheet(
                model, dataset, validation_indices, device, sheet_path,
                seed=args.seed + fold * 10000 + epoch)
            metrics = {
                "epoch": epoch + 1,
                "edge_temperature": temperature,
                "edge_weight": args.edge_weight,
                "trajectory_color_weight": args.trajectory_color_weight,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "contact_sheet": sheet_path.name,
            }
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            print("fold=%d epoch=%d/%d temperature=%.2f train=%.6f validation=%.6f" % (
                fold, epoch + 1, args.epochs, temperature, training_loss, validation_loss))
    torch.save(model.state_dict(), fold_dir / "model.pt")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-color-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    image_paths = sorted(args.data_dir.glob("*.png"))
    if len(image_paths) < FOLD_COUNT:
        raise ValueError("Need at least %d PNGs in %s" % (FOLD_COUNT, args.data_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RenderDataset(image_paths)
    device = mps_device()
    print("Training %d images with %d-fold cross-validation on %s" % (
        len(dataset), FOLD_COUNT, device))
    for fold, validation_indices in enumerate(fold_indices(len(dataset), args.seed), start=1):
        train_fold(dataset, validation_indices, fold, args, device)


if __name__ == "__main__":
    main()