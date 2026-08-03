"""Encoder-only LPE graph autoencoder for road-network GraphML files.

The encoder receives metadata features plus Laplacian positional encodings, then
uses learned attention pooling to produce one 256-dimensional graph embedding.
The decoder receives only that graph embedding broadcast to the graph's nodes
and the graph topology, so it must reconstruct metadata through message passing.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence, Optional

import networkx as nx
import torch
from pyproj import CRS, Transformer
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, LayerNorm
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.utils import softmax, scatter, degree


DEFAULT_NODE_FEATURES = (
    "start_y",
    "start_x",
    "reversed",
    "oneway",
    "length",
    "highway",
    "end_y",
    "end_x",
    "lanes",
    "maxspeed",
)

HIGHWAY_PRIORITY = (
    "footway", "service", "residential", "pedestrian", "tertiary", "secondary",
    "steps", "cycleway", "primary", "unclassified", "path", "corridor", "elevator",
    "platform", "busway", "motorway_link", "primary_link", "secondary_link", "motorway",
    "trunk", "track", "tertiary_link", "trunk_link", "living_street", "construction", "proposed",
)
HIGHWAY_INDEX = {highway: index for index, highway in enumerate(HIGHWAY_PRIORITY)}
TILE_IDENTIFIER = re.compile(r"--epsg(?P<epsg>\d+)--(?P<x>-?\d+)--(?P<y>-?\d+)\.graphml$")


def _first_number(value: object) -> float:
    """Extract a finite numeric value from GraphML metadata, or return zero."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else 0.0

    text = str(value).strip()
    try:
        result = float(text)
        return result if math.isfinite(result) else 0.0
    except ValueError:
        pass

    token = ""
    for character in text:
        if character.isdigit() or character in ".-+eE":
            token += character
        elif token:
            break
    try:
        result = float(token)
        return result if math.isfinite(result) else 0.0
    except ValueError:
        return 0.0


def _feature_number(feature: str, value: object) -> float:
    """Convert road metadata to its requested numeric representation."""
    text = str(value).strip() if value is not None else ""
    if feature in ("oneway", "reversed"):
        return float(text.lower() == "true")
    if feature == "highway":
        choices = re.findall(r"[a-zA-Z0-9_]+", text)
        return float(min((HIGHWAY_INDEX[choice] for choice in choices if choice in HIGHWAY_INDEX), default=0))

    number = _first_number(value)
    if feature == "maxspeed" and "mph" in text.lower():
        return number * 1.6
    if feature == "length" and text.lower().endswith("km"):
        return number * 1000.0
    return number


@lru_cache(maxsize=None)
def _tile_origin(path: Path) -> tuple[Transformer, float, float]:
    """Return the tile CRS transformer and its southwest corner in projected meters."""
    match = TILE_IDENTIFIER.search(path.name)
    if match is None:
        raise ValueError("Cannot determine EPSG and tile origin from %s" % path.name)
    epsg = int(match["epsg"])
    origin_x, origin_y = float(match["x"]), float(match["y"])
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
    return transformer, origin_x, origin_y


def _edge_index(graph: nx.Graph, nodes: Sequence[object]) -> Tensor:
    """Return a directed COO edge index with reverse links for GCN propagation."""
    node_indices = {node: index for index, node in enumerate(nodes)}
    edges = set()
    for source, target in graph.edges():
        source_index = node_indices[source]
        target_index = node_indices[target]
        edges.add((source_index, target_index))
        edges.add((target_index, source_index))

    if not edges:
        edges = {(index, index) for index in range(len(nodes))}
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()

def _add_lpe(x: Tensor, edge_index: Tensor, lpe_dim: int) -> Data:
    data = Data(x=x, edge_index=edge_index)
    if x.size(0) > 1 and lpe_dim > 0:
        available_lpe_dim = min(lpe_dim, x.size(0) - 1)
        data = AddLaplacianEigenvectorPE(
            k=available_lpe_dim,
            attr_name="lpe",
            is_undirected=True,
        )(data)
        if available_lpe_dim < lpe_dim:
            data.lpe = functional.pad(data.lpe, (0, lpe_dim - available_lpe_dim))
    else:
        data.lpe = torch.zeros((x.size(0), lpe_dim), dtype=torch.float32)
    return data.to(x.device)


def graphml_to_data(
    graph: nx.Graph,
    path: Path | None = None,
    feature_names: Sequence[str] = DEFAULT_NODE_FEATURES,
    lpe_dim: int = 8,
) -> Data:
    """Convert a NetworkX road graph to a PyG graph with numeric features and LPE."""
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot encode an empty graph")

    nodes = list(graph.nodes())
    transformer, origin_x, origin_y = _tile_origin(path) if path is not None else (None, 0.0, 0.0)

    def feature_value(node: object, feature: str) -> float:
        attributes = graph.nodes[node]
        if feature in ("start_x", "start_y", "end_x", "end_y") and transformer is not None:
            start_x, start_y = transformer.transform(
                _first_number(attributes.get("start_x")), _first_number(attributes.get("start_y")))
            end_x, end_y = transformer.transform(
                _first_number(attributes.get("end_x")), _first_number(attributes.get("end_y")))
            coordinates = {
                "start_x": start_x - origin_x,
                "start_y": start_y - origin_y,
                "end_x": end_x - origin_x,
                "end_y": end_y - origin_y,
            }
            return coordinates[feature]
        return _feature_number(feature, attributes.get(feature))

    x = torch.tensor(
        [[feature_value(node, feature) for feature in feature_names] for node in nodes],
        dtype=torch.float32,
    )
    edge_index = _edge_index(graph, nodes)
    data = _add_lpe(x, edge_index, lpe_dim)
    data = AddRandomWalkPE(walk_length=lpe_dim, attr_name="rwpe")(data)
    data.graph_path = str(path) if path is not None else ""
    return data


class TopologicalFiLMUnpool(nn.Module):
    def __init__(self, latent_dim: int = 256, k_steps: int = 8, hidden_dim: int = 128):
        super().__init__()
        
        # Input features: k_steps (RWPE from transform) + 1 (log degree)
        cond_dim = k_steps + 1
        
        # FiLM MLP: maps topological vector -> [gamma, beta]
        self.film_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

    def forward(self, z_graph: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, rwpe: torch.Tensor) -> torch.Tensor:
        # 1. Degree calculation (or pre-computed degree)
        deg = degree(edge_index[0], num_nodes=rwpe.size(0)).unsqueeze(-1)
        log_deg = torch.log(deg + 1.0)
        
        # 2. Combine pre-computed RWPE with log degree
        topo_cond = torch.cat([rwpe, log_deg], dim=-1)  # [N, k_steps + 1]
        
        # 3. Unpool z_graph to all nodes
        h_unpooled = z_graph[batch]                     # [N, latent_dim]
        
        # 4. Predict gamma and beta and modulate z_graph
        gamma_beta = self.film_mlp(topo_cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        return gamma * h_unpooled + beta


class TemperatureGlobalAttention(nn.Module):
    def __init__(self, gate_nn: nn.Module, value_nn: Optional[nn.Module] = None, temperature: float = 2.8):
        super().__init__()
        self.gate_nn = gate_nn
        self.value_nn = value_nn
        self.temperature = temperature

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, size: Optional[int] = None) -> torch.Tensor:
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        size = int(batch.max().item() + 1) if size is None and batch is not None else size

        # 1. Compute raw scores
        gate = self.gate_nn(x).view(-1, 1)

        # 2. Divide by temperature BEFORE softmax
        gate = softmax(gate / self.temperature, batch, num_nodes=size)

        # 3. Aggregate
        out = self.value_nn(x) if self.value_nn is not None else x
        return scatter(gate * out, batch, dim=0, dim_size=size, reduce='sum')

class RoadNetworkDataset(Dataset[Data]):
    """Load GraphML road networks and create a PyG graph for each valid file."""

    def __init__(
        self,
        graph_paths: Iterable[Path],
        feature_names: Sequence[str] = DEFAULT_NODE_FEATURES,
        lpe_dim: int = 8,
    ):
        self.graph_paths = list(graph_paths)
        if not self.graph_paths:
            raise ValueError("No GraphML files were provided")
        self.feature_names = tuple(feature_names)
        self.lpe_dim = lpe_dim

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, index: int) -> Data:
        path = self.graph_paths[index]
        return graphml_to_data(nx.read_graphml(path), path, self.feature_names, self.lpe_dim)


class FeatureStandardizer:
    """Fit per-feature normalization statistics and apply them to PyG graphs."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.mean = mean
        self.std = std.clamp_min(1e-6)

    @classmethod
    def fit(cls, dataset: Dataset[Data]) -> "FeatureStandardizer":
        features = torch.cat([dataset[index].x for index in range(len(dataset))], dim=0)
        return cls(features.mean(dim=0), features.std(dim=0, unbiased=False))

    def transform(self, data: Data) -> Data:
        data = data.clone()
        data.x = (data.x - self.mean) / self.std
        return data

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean, "std": self.std}


class EncoderOnlyLPEGAE(nn.Module):
    """Graph autoencoder with LPE only in the encoder and a 256-D graph code."""

    def __init__(
        self,
        in_channels: int,
        lpe_dim: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.lpe_proj = nn.Linear(lpe_dim, hidden_dim)
        self.enc_gcn1 = GCNConv(in_channels + hidden_dim, hidden_dim)
        self.enc_gcns = nn.ModuleList([GATConv(hidden_dim, hidden_dim, heads=4, concat=False) for i in range(num_layers)])
        self.enc_norm = nn.ModuleList([LayerNorm(hidden_dim) for i in range(num_layers)])
        self.enc_gcnF = GCNConv(hidden_dim, hidden_dim)
        self.pool_lpe_proj = nn.Linear(lpe_dim, hidden_dim)
        self.pooling = TemperatureGlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        )
        self.latent_proj = nn.Linear(hidden_dim * 2, latent_dim)

        self.dec_unpool = TopologicalFiLMUnpool(latent_dim=latent_dim, k_steps=lpe_dim, hidden_dim=hidden_dim)
        self.dec_gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.dec_gcns = nn.ModuleList([GATConv(hidden_dim, hidden_dim, heads=4, concat=False) for i in range(num_layers)])
        self.dec_norm = nn.ModuleList([LayerNorm(hidden_dim) for i in range(num_layers)]) 
        self.dec_gcnF = GCNConv(hidden_dim, in_channels)
        
        self.activation = lambda x: functional.leaky_relu(x, negative_slope=0.1)

    def encode(self, x: Tensor, lpe: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Encode batched nodes into one latent vector per graph."""
        if self.training:
            sign_flip = torch.randint(0, 2, (1, lpe.size(-1)), device=lpe.device, dtype=torch.long)
            lpe = lpe * (sign_flip.mul(2).sub(1).to(lpe.dtype))
        lpe_embedding = self.activation(self.lpe_proj(lpe))
        hidden = self.activation(self.enc_gcn1(torch.cat((x, lpe_embedding), dim=-1), edge_index))
       
        for gcn, norm in zip(self.enc_gcns, self.enc_norm):
            h_norm = norm(hidden, batch)
            h_conv = self.activation(gcn(h_norm, edge_index))
            hidden = h_norm + h_conv
        latent = self.enc_gcnF(hidden, edge_index)
        latent = _add_lpe(latent, edge_index, self.lpe_proj.in_features)
        lpe_pool_embeddings = self.activation(self.pool_lpe_proj(latent.lpe))
        latent = torch.cat((latent, lpe_pool_embeddings), dim=-1)
        return self.latent_proj(self.pooling(latent, batch))

    def decode(self, z_graph: Tensor, rwpe: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Reconstruct node metadata from graph codes and topology, without LPE."""
        h = self.dec_unpool(z_graph, edge_index, batch, rwpe)
        hidden = self.activation(self.dec_gcn1(h, edge_index))
        
        
        for gcn, norm in zip(self.dec_gcns, self.dec_norm):
            h_norm = norm(hidden, batch)
            h_conv = self.activation(gcn(h_norm, edge_index))
            hidden = h_norm + h_conv
 
        return self.dec_gcnF(hidden, edge_index)

    def forward(self, x: Tensor, lpe: Tensor, rwpe: Tensor, edge_index: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        graph_embeddings = self.encode(x, lpe, edge_index, batch)
        return self.decode(graph_embeddings, rwpe, edge_index, batch), graph_embeddings


def training_metadata(feature_names: Sequence[str], lpe_dim: int) -> str:
    """Return serializable configuration metadata for saved checkpoints."""
    return json.dumps({"feature_names": list(feature_names), "lpe_dim": lpe_dim})
