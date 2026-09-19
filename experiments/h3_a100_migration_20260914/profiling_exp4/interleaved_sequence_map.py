from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterleavedSequenceMap:
    """Logical/physical index map for rank-interleaved sequence sharding.

    ``physical`` order is the rank-concatenated order produced by Ulysses:
    rank 0 owns logical ``0, world_size, ...``, rank 1 owns logical
    ``1, world_size + 1, ...``, and so on.
    """

    length: int
    world_size: int

    def __post_init__(self) -> None:
        if self.length <= 0 or self.world_size <= 0:
            raise ValueError("length and world_size must be positive")
        if self.length % self.world_size:
            raise ValueError("length must be divisible by world_size")

    @property
    def local_rows(self) -> int:
        return self.length // self.world_size

    def logical_to_physical(self, logical: int) -> int:
        if not 0 <= logical < self.length:
            raise IndexError(logical)
        owner = logical % self.world_size
        local_index = logical // self.world_size
        return owner * self.local_rows + local_index

    def physical_to_logical(self, physical: int) -> int:
        if not 0 <= physical < self.length:
            raise IndexError(physical)
        owner = physical // self.local_rows
        local_index = physical % self.local_rows
        return local_index * self.world_size + owner

    def logical_indices_for_rank(self, rank: int) -> range:
        if not 0 <= rank < self.world_size:
            raise IndexError(rank)
        return range(rank, self.length, self.world_size)

