from __future__ import annotations

import unittest

from interleaved_sequence_map import InterleavedSequenceMap


class InterleavedSequenceMapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = InterleavedSequenceMap(length=81216, world_size=2)

    def test_known_cross_shard_neighbours(self) -> None:
        # First two spatial neighbours in target interior frame 1.
        self.assertEqual(self.mapping.logical_to_physical(44463), 62839)
        self.assertEqual(self.mapping.logical_to_physical(44464), 22232)
        self.assertEqual(self.mapping.physical_to_logical(62839), 44463)
        self.assertEqual(self.mapping.physical_to_logical(22232), 44464)

        # Reusing logical indices as physical indices is the historical bug.
        self.assertEqual(self.mapping.physical_to_logical(44463), 7711)
        self.assertEqual(self.mapping.physical_to_logical(44464), 7713)

    def test_segment_and_padding_boundaries_round_trip(self) -> None:
        # text/source, source/target, target/padding, and final padded row.
        boundaries = [0, 6158, 6159, 43454, 43455, 80750, 80751, 81215]
        for logical in boundaries:
            with self.subTest(logical=logical):
                physical = self.mapping.logical_to_physical(logical)
                self.assertEqual(self.mapping.physical_to_logical(physical), logical)

    def test_rank_ownership_is_interleaved(self) -> None:
        self.assertEqual(list(self.mapping.logical_indices_for_rank(0))[:4], [0, 2, 4, 6])
        self.assertEqual(list(self.mapping.logical_indices_for_rank(1))[:4], [1, 3, 5, 7])
        self.assertEqual(len(self.mapping.logical_indices_for_rank(0)), 40608)
        self.assertEqual(len(self.mapping.logical_indices_for_rank(1)), 40608)


if __name__ == "__main__":
    unittest.main()
