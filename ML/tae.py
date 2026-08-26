"""Bidirectional-LSTM trajectory autoencoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


FEATURE_DIMENSIONS = 3
EMBEDDING_DIMENSIONS = 128


class TrajectoryAutoencoder(nn.Module):
    """Encode variable-length UTM/time trajectories into 128 values and reconstruct them."""

    def __init__(
        self,
        feature_dimensions: int = FEATURE_DIMENSIONS,
        embedding_dimensions: int = EMBEDDING_DIMENSIONS,
        hidden_dimensions: int = 128,
    ) -> None:
        super().__init__()
        self.feature_dimensions = feature_dimensions
        self.embedding_dimensions = embedding_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.encoder = nn.LSTM(
            input_size=feature_dimensions,
            hidden_size=hidden_dimensions,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.to_embedding = nn.Linear(2 * hidden_dimensions, embedding_dimensions)
        self.decoder = nn.LSTM(
            input_size=embedding_dimensions + 1,
            hidden_size=hidden_dimensions,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.to_features = nn.Linear(2 * hidden_dimensions, feature_dimensions)

    def encode(self, trajectories: Tensor, lengths: Tensor) -> Tensor:
        """Return one embedding per padded trajectory through a dense LSTM pass."""
        self._validate_inputs(trajectories, lengths)
        outputs, _ = self.encoder(trajectories)
        batch_indices = torch.arange(trajectories.shape[0], device=trajectories.device)
        last_forward = outputs[batch_indices, lengths - 1, :self.hidden_dimensions]
        first_backward = outputs[:, 0, self.hidden_dimensions:]
        return self.to_embedding(torch.cat((last_forward, first_backward), dim=1))

    def decode(self, embeddings: Tensor, lengths: Tensor, sequence_length: int) -> Tensor:
        """Reconstruct padded sequences, using normalized position as the decoder input."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dimensions:
            raise ValueError("Expected embeddings shaped (batch, embedding_dimensions)")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        positions = torch.arange(
            sequence_length, device=embeddings.device, dtype=embeddings.dtype).unsqueeze(0)
        positions = positions.expand(embeddings.shape[0], -1)
        positions = positions / (lengths.to(embeddings.dtype).unsqueeze(1) - 1).clamp_min(1)
        decoder_inputs = torch.cat((
            embeddings.unsqueeze(1).expand(-1, sequence_length, -1),
            positions.unsqueeze(-1),
        ), dim=-1)
        outputs, _ = self.decoder(decoder_inputs)
        return self.to_features(outputs)

    def forward(self, trajectories: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Return padded reconstructions and their corresponding embeddings."""
        embeddings = self.encode(trajectories, lengths)
        return self.decode(embeddings, lengths, trajectories.shape[1]), embeddings

    def _validate_inputs(self, trajectories: Tensor, lengths: Tensor) -> None:
        if trajectories.ndim != 3 or trajectories.shape[2] != self.feature_dimensions:
            raise ValueError("Expected trajectories shaped (batch, sequence, feature_dimensions)")
        if lengths.ndim != 1 or lengths.shape[0] != trajectories.shape[0]:
            raise ValueError("Expected one sequence length per trajectory")
        if lengths.device != trajectories.device:
            raise ValueError("Trajectory lengths and points must be on the same device")


def reconstruction_loss(reconstruction: Tensor, targets: Tensor, valid_points: Tensor) -> Tensor:
    """Return vectorized MSE over valid points without dynamic boolean indexing."""
    if reconstruction.shape != targets.shape:
        raise ValueError("Reconstruction and target shapes must match")
    if valid_points.shape != targets.shape[:2] + (1,):
        raise ValueError("Expected valid_points shaped (batch, sequence, 1)")
    squared_error = (reconstruction - targets).square()
    masked_error = squared_error * valid_points
    valid_element_count = valid_points.sum() * targets.shape[-1]
    return masked_error.sum() / valid_element_count.clamp_min(1)