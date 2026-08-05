"""Tensor-level tests for macro tracking and constrained road-link decoding."""

import unittest

import torch

from ML.sequence_decoder import DualHeadDecoder, MacroSequenceTracker


class SequenceDecoderTest(unittest.TestCase):
    def test_tracker_respects_padded_macro_step_lengths(self):
        tracker = MacroSequenceTracker()
        vectors = tracker(torch.randn(2, 3, 643), torch.tensor([3, 2]))
        self.assertEqual(vectors.shape, (2, 3, 1024))

    def test_teacher_forcing_trains_end_token_after_maximum_length_target(self):
        decoder = DualHeadDecoder()
        macro_vectors = torch.randn(1, 1, 1024)
        candidate_embeddings = torch.randn(1, 1, 3, 256)
        adjacency = [[{1: [1, 2], 2: [1, 2]}]]
        target_links = torch.tensor([[[1, 2]]])
        target_lengths = torch.tensor([[2]])
        target_times = torch.tensor([[[0.3, 0.8]]])

        logits, predicted_times, spatial_valid, temporal_valid = decoder.decode_teacher_forcing(
            macro_vectors, candidate_embeddings, adjacency, target_links, target_lengths)

        self.assertEqual(logits.shape, (1, 1, 3, 3))
        self.assertEqual(spatial_valid.tolist(), [[[True, True, True]]])
        self.assertEqual(temporal_valid.tolist(), [[[True, True, False]]])
        self.assertLess(predicted_times[0, 0, 0], predicted_times[0, 0, 1])
        loss, _ = decoder.teacher_forcing_loss(
            macro_vectors, candidate_embeddings, adjacency, target_links, target_lengths, target_times)
        loss.backward()


if __name__ == "__main__":
    unittest.main()