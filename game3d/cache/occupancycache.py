from __future__ import annotations
#game3d/cache/occupancycache.py
"""Zero-copy occupancy cache – boolean mask (9,9,9)."""

from typing import Tuple, Optional
import torch
from game3d.board.board import Board


class OccupancyCache:
    __slots__ = ("mask",)

    def __init__(self, board: Board) -> None:
        self.mask: torch.Tensor  # (9, 9, 9), dtype=torch.bool
        self.rebuild(board)

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """O(1) scalar query."""
        # 🔥 Optimization: avoid .item() if possible (but safe here)
        return bool(self.mask[z, y, x])

    def is_occupied_batch(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N,3) long tensor with columns (x, y, z)
        returns: (N,) bool tensor
        """
        # Ensure coords are long/int64 for indexing
        return self.mask[coords[:, 2], coords[:, 1], coords[:, 0]]

    @property
    def count(self) -> int:
        """Total occupied squares."""
        return int(self.mask.sum().item())

    def rebuild(self, board: Board) -> None:
        """
        Zero-copy view – shares memory with board's internal occupancy plane.
        ⚠️ Assumes board.occupancy_mask() returns a view, not a copy.
        """
        mask = board.occupancy_mask()
        if mask.dtype != torch.bool or mask.shape != (9, 9, 9):
            raise ValueError("Board must provide (9,9,9) bool occupancy mask")
        self.mask = mask


