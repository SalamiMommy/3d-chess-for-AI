from __future__ import annotations
# game3d/cache/occupancycache.py
"""Zero-copy occupancy cache – boolean mask (9,9,9)."""

from typing import Tuple, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from game3d.board.board import Board
else:
    Board = object  # Dummy type for runtime

class OccupancyCache:
    __slots__ = ("mask",)

    def __init__(self, board: "Board") -> None:
        self.mask: torch.Tensor  # (9, 9, 9), dtype=torch.bool
        self.rebuild(board)

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """O(1) scalar query."""
        return bool(self.mask[z, y, x])

    def is_occupied_batch(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mask[coords[:, 2], coords[:, 1], coords[:, 0]]

    @property
    def count(self) -> int:
        return int(self.mask.sum().item())

    def rebuild(self, board: "Board") -> None:
        mask = board.occupancy_mask()
        if mask.dtype != torch.bool or mask.shape != (9, 9, 9):
            raise ValueError("Board must provide (9,9,9) bool occupancy mask")
        self.mask = mask


