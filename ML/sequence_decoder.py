"""Macro-step tracking and topologically constrained road-link decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


LOCAL_FUSION_DIMENSIONS = 643
GCN_EMBEDDING_DIMENSIONS = 256
MACRO_VECTOR_DIMENSIONS = 1024


class MacroSequenceTracker(nn.Module):
    """Resolve local macro-step dynamics and global sequence context."""

    def __init__(self) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=LOCAL_FUSION_DIMENSIONS,
            hidden_size=MACRO_VECTOR_DIMENSIONS // 2,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MACRO_VECTOR_DIMENSIONS,
            nhead=2,
            dim_feedforward=2048,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, fused_steps: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Return ``[batch, macro_steps, 1024]`` vectors from 643-value inputs."""
        if fused_steps.ndim != 3 or fused_steps.shape[-1] != LOCAL_FUSION_DIMENSIONS:
            raise ValueError("Expected fused_steps shaped (batch, macro_steps, 643)")
        padding_mask = None
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != fused_steps.shape[0]:
                raise ValueError("Expected one macro-step length per batch item")
            if lengths.device != fused_steps.device:
                raise ValueError("Macro-step lengths and fused_steps must share a device")
            if torch.any(lengths < 1) or torch.any(lengths > fused_steps.shape[1]):
                raise ValueError("Macro-step lengths must be within the padded sequence")
            packed = nn.utils.rnn.pack_padded_sequence(
                fused_steps, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_outputs, _ = self.bilstm(packed)
            lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outputs, batch_first=True, total_length=fused_steps.shape[1])
            step_indices = torch.arange(fused_steps.shape[1], device=fused_steps.device).unsqueeze(0)
            padding_mask = step_indices >= lengths.unsqueeze(1)
        else:
            lstm_outputs, _ = self.bilstm(fused_steps)
        return self.transformer(lstm_outputs, src_key_padding_mask=padding_mask)


class DualHeadDecoder(nn.Module):
    """Autoregressively select connected road links and their cumulative times."""

    def __init__(self) -> None:
        super().__init__()
        self.start_token = nn.Parameter(torch.empty(GCN_EMBEDDING_DIMENSIONS))
        nn.init.normal_(self.start_token, std=0.02)
        self.decoder_cell = nn.GRUCell(
            input_size=MACRO_VECTOR_DIMENSIONS + GCN_EMBEDDING_DIMENSIONS,
            hidden_size=MACRO_VECTOR_DIMENSIONS,
        )
        self.pointer = nn.Linear(MACRO_VECTOR_DIMENSIONS, GCN_EMBEDDING_DIMENSIONS, bias=False)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(MACRO_VECTOR_DIMENSIONS, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def decode_teacher_forcing(
        self,
        macro_vectors: Tensor,
        local_gcn_embeddings: Tensor,
        adjacency: Sequence[Mapping[int, Sequence[int]]],
        target_links: Tensor,
        target_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode batched targets, feeding each ground-truth link into the next step.

        ``local_gcn_embeddings`` is shaped ``[batch, macro_steps, candidates, 256]``
        and reserves candidate index zero for ``<END>``. ``target_links`` contains
        nonzero candidate indices and is shaped ``[batch, macro_steps, sub_steps]``;
        ``target_lengths`` gives its valid non-END link count per macro-step.
        """
        batch_size, macro_steps, candidate_count = self._validate_teacher_inputs(
            macro_vectors, local_gcn_embeddings, adjacency, target_links, target_lengths)
        target_sub_steps = target_links.shape[-1]
        decode_steps = target_sub_steps + 1
        flat_count = batch_size * macro_steps
        macro_vectors = macro_vectors.reshape(flat_count, MACRO_VECTOR_DIMENSIONS)
        candidates = local_gcn_embeddings.reshape(flat_count, candidate_count, GCN_EMBEDDING_DIMENSIONS)
        target_links = target_links.reshape(flat_count, target_sub_steps)
        target_lengths = target_lengths.reshape(flat_count)
        flat_adjacency = [item for macro_adjacency in adjacency for item in macro_adjacency]

        previous_embeddings = self.start_token.expand(flat_count, -1)
        previous_links = torch.full((flat_count,), -1, dtype=torch.long, device=macro_vectors.device)
        hidden_states = macro_vectors
        spatial_logits = []
        cumulative_predictions = []
        spatial_valid = []
        temporal_valid = []
        current_times = torch.zeros(flat_count, dtype=macro_vectors.dtype, device=macro_vectors.device)

        for step in range(decode_steps):
            hidden_states = self.decoder_cell(torch.cat((macro_vectors, previous_embeddings), dim=-1), hidden_states)
            logits = self._masked_pointer_logits(hidden_states, candidates, previous_links, flat_adjacency)
            progression = self.temporal_mlp(hidden_states).squeeze(-1)
            current_times = current_times + (1.0 - current_times) * progression
            spatial_logits.append(logits)
            cumulative_predictions.append(current_times)
            spatial_valid.append(step <= target_lengths)
            temporal_valid.append(step < target_lengths)

            target_link = target_links[:, step] if step < target_sub_steps else torch.zeros_like(previous_links)
            previous_links = torch.where(
                step < target_lengths, target_link, torch.zeros_like(target_link))
            previous_embeddings = candidates[torch.arange(flat_count, device=macro_vectors.device), previous_links]

        return (
            torch.stack(spatial_logits, dim=1).reshape(batch_size, macro_steps, decode_steps, candidate_count),
            torch.stack(cumulative_predictions, dim=1).reshape(batch_size, macro_steps, decode_steps),
            torch.stack(spatial_valid, dim=1).reshape(batch_size, macro_steps, decode_steps),
            torch.stack(temporal_valid, dim=1).reshape(batch_size, macro_steps, decode_steps),
        )

    def teacher_forcing_loss(
        self,
        macro_vectors: Tensor,
        local_gcn_embeddings: Tensor,
        adjacency: Sequence[Mapping[int, Sequence[int]]],
        target_links: Tensor,
        target_lengths: Tensor,
        target_cumulative_times: Tensor,
        temporal_loss: str = "huber",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return masked pointer and cumulative-time losses under teacher forcing."""
        if target_cumulative_times.shape != target_links.shape:
            raise ValueError("target_cumulative_times and target_links must have matching shapes")
        logits, predicted_times, spatial_valid, temporal_valid = self.decode_teacher_forcing(
            macro_vectors, local_gcn_embeddings, adjacency, target_links, target_lengths)
        end_targets = torch.zeros_like(target_links[..., :1])
        spatial_targets = torch.cat((target_links, end_targets), dim=-1)
        spatial_targets = torch.where(temporal_valid, spatial_targets, torch.zeros_like(spatial_targets))
        spatial_loss = functional.cross_entropy(logits[spatial_valid], spatial_targets[spatial_valid])
        if not temporal_valid.any():
            raise ValueError("At least one target road link is required for temporal supervision")
        time_targets = torch.cat((target_cumulative_times, torch.zeros_like(target_cumulative_times[..., :1])), dim=-1)
        if temporal_loss == "huber":
            time_loss = functional.huber_loss(predicted_times[temporal_valid], time_targets[temporal_valid])
        elif temporal_loss == "mse":
            time_loss = functional.mse_loss(predicted_times[temporal_valid], time_targets[temporal_valid])
        else:
            raise ValueError("temporal_loss must be 'huber' or 'mse'")
        total_loss = spatial_loss + time_loss
        return total_loss, {"spatial_loss": spatial_loss, "temporal_loss": time_loss}

    @torch.no_grad()
    def decode_macro_step(
        self,
        macro_vector: Tensor,
        local_gcn_embeddings: Tensor,
        adjacency: Mapping[int, Sequence[int]],
        max_sub_steps: int = 10,
    ) -> tuple[list[int], list[float]]:
        """Greedily decode one macro vector using a hard downstream-link mask."""
        if macro_vector.shape != (MACRO_VECTOR_DIMENSIONS,):
            raise ValueError("Expected macro_vector shaped (1024,)")
        if local_gcn_embeddings.ndim != 2 or local_gcn_embeddings.shape[1] != GCN_EMBEDDING_DIMENSIONS:
            raise ValueError("Expected local_gcn_embeddings shaped (candidates, 256)")
        if local_gcn_embeddings.shape[0] < 2 or max_sub_steps <= 0:
            raise ValueError("At least <END> and one road candidate are required; max_sub_steps must be positive")
        hidden_state = macro_vector
        previous_embedding = self.start_token
        previous_link = -1
        current_time = 0.0
        links: list[int] = []
        cumulative_times: list[float] = []
        for _ in range(max_sub_steps):
            hidden_state = self.decoder_cell(torch.cat((macro_vector, previous_embedding)), hidden_state)
            logits = self._masked_pointer_logits(
                hidden_state.unsqueeze(0), local_gcn_embeddings.unsqueeze(0),
                torch.tensor([previous_link], device=macro_vector.device), [adjacency])[0]
            selected_link = int(logits.argmax().item())
            if selected_link == 0:
                break
            progression = self.temporal_mlp(hidden_state).item()
            current_time = current_time + (1.0 - current_time) * progression
            links.append(selected_link)
            cumulative_times.append(current_time)
            previous_link = selected_link
            previous_embedding = local_gcn_embeddings[selected_link]
        return links, cumulative_times

    def _masked_pointer_logits(
        self,
        hidden_states: Tensor,
        candidates: Tensor,
        previous_links: Tensor,
        adjacency: Sequence[Mapping[int, Sequence[int]]],
    ) -> Tensor:
        logits = torch.bmm(candidates, self.pointer(hidden_states).unsqueeze(-1)).squeeze(-1)
        mask = torch.full_like(logits, float("-inf"))
        mask[:, 0] = 0.0
        for index, previous_link in enumerate(previous_links.tolist()):
            valid_links = range(1, candidates.shape[1]) if previous_link < 0 else adjacency[index].get(previous_link, ())
            for valid_link in valid_links:
                if 0 < valid_link < candidates.shape[1]:
                    mask[index, valid_link] = 0.0
        return logits + mask

    @staticmethod
    def _validate_teacher_inputs(
        macro_vectors: Tensor,
        local_gcn_embeddings: Tensor,
        adjacency: Sequence[Mapping[int, Sequence[int]]],
        target_links: Tensor,
        target_lengths: Tensor,
    ) -> tuple[int, int, int]:
        if macro_vectors.ndim != 3 or macro_vectors.shape[-1] != MACRO_VECTOR_DIMENSIONS:
            raise ValueError("Expected macro_vectors shaped (batch, macro_steps, 1024)")
        batch_size, macro_steps, _ = macro_vectors.shape
        expected_candidates = (batch_size, macro_steps)
        if local_gcn_embeddings.ndim != 4 or local_gcn_embeddings.shape[:2] != expected_candidates:
            raise ValueError("Expected local_gcn_embeddings shaped (batch, macro_steps, candidates, 256)")
        if local_gcn_embeddings.shape[-1] != GCN_EMBEDDING_DIMENSIONS or local_gcn_embeddings.shape[2] < 2:
            raise ValueError("Expected at least two 256-dimensional candidates per macro-step")
        if target_links.ndim != 3 or target_links.shape[:2] != expected_candidates:
            raise ValueError("Expected target_links shaped (batch, macro_steps, sub_steps)")
        if target_lengths.shape != expected_candidates:
            raise ValueError("Expected target_lengths shaped (batch, macro_steps)")
        if len(adjacency) != batch_size or any(len(item) != macro_steps for item in adjacency):
            raise ValueError("Expected one adjacency mapping per batch item and macro-step")
        if target_links.device != macro_vectors.device or local_gcn_embeddings.device != macro_vectors.device:
            raise ValueError("Decoder tensors must share a device")
        if torch.any(target_lengths < 0) or torch.any(target_lengths > target_links.shape[-1]):
            raise ValueError("target_lengths must be within the target sub-step sequence")
        if torch.any(target_links < 0) or torch.any(target_links >= local_gcn_embeddings.shape[2]):
            raise ValueError("target_links must be valid candidate indices")
        return batch_size, macro_steps, local_gcn_embeddings.shape[2]