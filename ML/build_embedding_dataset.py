#!/usr/bin/env python3
"""Build matched multimodal encoder embeddings keyed by exact tile ID.

The builder discovers ``cae_model.pt``, ``tae_model.pt``, and ``gae_model.pt``
in ``models/``. It encodes every available source modality, joins only tile IDs
present in every available modality, and writes a CPU-portable ``.pt`` dataset.
When a GAE checkpoint is available, each entry also retains the final
pre-pooling node embeddings aligned with the original GraphML node identifiers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from PIL import Image
from pyproj import Transformer

try:
    from .cae import CoordinateAutoencoder
    from .gae import DEFAULT_NODE_FEATURES, EncoderOnlyLPEGAE, FeatureStandardizer as GAEFeatureStandardizer, graphml_to_data
    from .tae import TrajectoryAutoencoder
except ImportError:
    from cae import CoordinateAutoencoder
    from gae import DEFAULT_NODE_FEATURES, EncoderOnlyLPEGAE, FeatureStandardizer as GAEFeatureStandardizer, graphml_to_data
    from tae import TrajectoryAutoencoder


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPOSITORY_ROOT / "models"
DEFAULT_TRAJECTORIES_PATH = REPOSITORY_ROOT / "datasets" / "trajectories" / "trajectory_samples.jsonl"
DEFAULT_RENDERS_DIR = REPOSITORY_ROOT / "datasets" / "standard_map_renders"
DEFAULT_ROAD_NETWORKS_DIR = REPOSITORY_ROOT / "datasets" / "road_networks"
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "datasets" / "encoder_embeddings" / "matched_embeddings.pt"


def accelerator_device(requested_device: str) -> torch.device:
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if requested_device != "auto":
        if not available[requested_device]:
            raise RuntimeError("Requested %s accelerator is unavailable" % requested_device)
        return torch.device(requested_device)
    return torch.device(next(name for name in ("cuda", "mps", "cpu") if available[name]))


def load_state_dict(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary in %s" % path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state_dict.items()):
        raise ValueError("Checkpoint %s does not contain a model state dict" % path)
    return state_dict, checkpoint


def load_cae(path: Path, device: torch.device) -> CoordinateAutoencoder:
    state_dict, _ = load_state_dict(path)
    model = CoordinateAutoencoder().to(device)
    model.load_state_dict(state_dict)
    return model.eval()


def load_tae(path: Path, device: torch.device) -> tuple[TrajectoryAutoencoder, dict[str, torch.Tensor]]:
    state_dict, checkpoint = load_state_dict(path)
    config = checkpoint.get("model_config", {})
    model = TrajectoryAutoencoder(
        feature_dimensions=config.get("feature_dimensions", 3),
        embedding_dimensions=config.get("embedding_dimensions", 128),
        hidden_dimensions=config.get("hidden_dimensions", 128),
    ).to(device)
    model.load_state_dict(state_dict)
    standardizer = checkpoint.get("feature_standardizer")
    if not isinstance(standardizer, dict) or set(standardizer) != {"mean", "standard_deviation"}:
        raise ValueError("TAE checkpoint %s must include feature_standardizer statistics" % path)
    return model.eval(), standardizer


def load_gae(path: Path, device: torch.device) -> tuple[EncoderOnlyLPEGAE, GAEFeatureStandardizer, dict[str, Any]]:
    state_dict, checkpoint = load_state_dict(path)
    config = checkpoint.get("model_config", {})
    model = EncoderOnlyLPEGAE(
        in_channels=config.get("in_channels", len(DEFAULT_NODE_FEATURES)),
        lpe_dim=config.get("lpe_dim", 8),
        hidden_dim=config.get("hidden_dim", 128),
        latent_dim=config.get("latent_dim", 256),
        num_layers=config.get("num_layers", 3),
    ).to(device)
    model.load_state_dict(state_dict)
    standardizer_state = checkpoint.get("feature_standardizer")
    if not isinstance(standardizer_state, dict) or set(standardizer_state) != {"mean", "std"}:
        raise ValueError("GAE checkpoint %s must include feature_standardizer statistics" % path)
    return model.eval(), GAEFeatureStandardizer(**standardizer_state), config


def trajectory_tensor(entry: dict[str, Any], transformer_cache: dict[int, Transformer]) -> torch.Tensor:
    points = entry["tile_points"]
    if not points:
        raise ValueError("Trajectory %s has no tile points" % entry["tile_id"])
    epsg = int(entry["epsg"])
    transformer = transformer_cache.setdefault(
        epsg, Transformer.from_crs("EPSG:4326", "EPSG:%d" % epsg, always_xy=True))
    southwest_easting, southwest_northing = entry["bounds_utm_m"][:2]
    start_timestamp = points[0]["ts"]
    return torch.tensor([
        (
            transformer.transform(point["longitude"], point["latitude"])[0] - southwest_easting,
            transformer.transform(point["longitude"], point["latitude"])[1] - southwest_northing,
            point["ts"] - start_timestamp,
        )
        for point in points
    ], dtype=torch.float32)


def encode_trajectories(
    data_path: Path, model: TrajectoryAutoencoder, standardizer: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    transformers: dict[int, Transformer] = {}
    mean = standardizer["mean"]
    standard_deviation = standardizer["standard_deviation"].clamp_min(1e-6)
    with data_path.open() as source, torch.inference_mode():
        for line in source:
            entry = json.loads(line)
            tile_id = entry["tile_id"]
            if tile_id in embeddings:
                raise ValueError("Duplicate trajectory tile_id %s" % tile_id)
            trajectory = trajectory_tensor(entry, transformers)
            normalized = ((trajectory - mean) / standard_deviation).unsqueeze(0).to(device)
            length = torch.tensor([len(trajectory)], device=device)
            embeddings[tile_id] = model.encode(normalized, length).squeeze(0).cpu()
    return embeddings


def encode_renders(render_paths: list[Path], model: CoordinateAutoencoder, device: torch.device) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for path in render_paths:
            if path.stem in embeddings:
                raise ValueError("Duplicate render tile_id %s" % path.stem)
            pixels = torch.from_numpy(__import__("numpy").asarray(Image.open(path).convert("RGB")).copy())
            image = pixels.permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            embeddings[path.stem] = model.encode(image)[0].squeeze(0).cpu()
    return embeddings


def encode_gae_graph(model: EncoderOnlyLPEGAE, data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final graph code and the encoder output immediately before pooling."""
    data = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    lpe_embedding = model.activation(model.lpe_proj(data.lpe))
    lpe_embedding = model.enc_rwpe_film(lpe_embedding, data.rwpe, data.edge_index)
    hidden = model.activation(model.enc_gcn1(torch.cat((data.x, lpe_embedding), dim=-1), data.edge_index))
    for gcn, norm in zip(model.enc_gcns, model.enc_norm):
        normalized = norm(hidden, batch)
        hidden = normalized + model.activation(gcn(normalized, data.edge_index))
    node_embeddings = model.enc_gcnF(hidden, data.edge_index)
    pool_lpe_embeddings = model.activation(model.pool_lpe_proj(data.lpe))
    graph_embedding = model.latent_proj(model.pooling(torch.cat((node_embeddings, pool_lpe_embeddings), dim=-1), batch))
    return graph_embedding.squeeze(0).cpu(), node_embeddings.cpu()


def encode_road_networks(
    graph_paths: list[Path], model: EncoderOnlyLPEGAE, standardizer: GAEFeatureStandardizer,
    lpe_dim: int, device: torch.device,
) -> dict[str, dict[str, Any]]:
    embeddings: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for path in graph_paths:
            tile_id = path.stem
            if tile_id in embeddings:
                raise ValueError("Duplicate road-network tile_id %s" % tile_id)
            graph = nx.read_graphml(path)
            node_ids = [str(node_id) for node_id in graph.nodes()]
            data = standardizer.transform(graphml_to_data(graph, path=path, lpe_dim=lpe_dim))
            graph_embedding, node_embeddings = encode_gae_graph(model, data, device)
            embeddings[tile_id] = {
                "embedding": graph_embedding,
                "node_embeddings": node_embeddings,
                "node_ids": node_ids,
                "road_network_path": str(path.relative_to(REPOSITORY_ROOT)),
            }
    return embeddings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--trajectories-path", type=Path, default=DEFAULT_TRAJECTORIES_PATH)
    parser.add_argument("--renders-dir", type=Path, default=DEFAULT_RENDERS_DIR)
    parser.add_argument("--road-networks-dir", type=Path, default=DEFAULT_ROAD_NETWORKS_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--require-gae", action="store_true", help="Fail unless models/gae_model.pt exists")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = accelerator_device(args.device)
    cae_path = args.models_dir / "cae_model.pt"
    tae_path = args.models_dir / "tae_model.pt"
    gae_path = args.models_dir / "gae_model.pt"
    missing = [str(path) for path in (cae_path, tae_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required encoder checkpoints: %s" % ", ".join(missing))
    if args.require_gae and not gae_path.exists():
        raise FileNotFoundError("Missing required GAE checkpoint: %s" % gae_path)

    cae = load_cae(cae_path, device)
    tae, tae_standardizer = load_tae(tae_path, device)
    trajectory_embeddings = encode_trajectories(args.trajectories_path, tae, tae_standardizer, device)
    render_embeddings = encode_renders(sorted(args.renders_dir.glob("*.png")), cae, device)
    modalities: dict[str, set[str]] = {"trajectory": set(trajectory_embeddings), "render": set(render_embeddings)}
    road_embeddings = None
    if gae_path.exists():
        gae, gae_standardizer, gae_config = load_gae(gae_path, device)
        road_embeddings = encode_road_networks(
            sorted(args.road_networks_dir.glob("*.graphml")), gae, gae_standardizer,
            gae_config.get("lpe_dim", 8), device)
        modalities["road_network"] = set(road_embeddings)

    matched_ids = sorted(set.intersection(*modalities.values()))
    if not matched_ids:
        raise ValueError("No tile IDs are shared by all available modalities")
    entries = []
    for tile_id in matched_ids:
        entry = {"tile_id": tile_id, "tae_embedding": trajectory_embeddings[tile_id], "cae_embedding": render_embeddings[tile_id]}
        if road_embeddings is not None:
            entry["gae_embedding"] = road_embeddings[tile_id]["embedding"]
            entry["gae_node_embeddings"] = road_embeddings[tile_id]["node_embeddings"]
            entry["gae_node_ids"] = road_embeddings[tile_id]["node_ids"]
            entry["road_network_path"] = road_embeddings[tile_id]["road_network_path"]
        entries.append(entry)
    report = {
        "available_modalities": sorted(modalities),
        "source_counts": {name: len(tile_ids) for name, tile_ids in modalities.items()},
        "matched_count": len(entries),
        "unmatched_counts": {name: len(tile_ids - set(matched_ids)) for name, tile_ids in modalities.items()},
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"entries": entries, "report": report}, args.output_path)
    print("Saved %d matched embedding entries to %s" % (len(entries), args.output_path))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()