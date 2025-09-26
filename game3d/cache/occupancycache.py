"""Zero-copy occupancy cache – boolean mask (9,9,9)."""
#game3d/cache/occupancycache.py
from __future__ import annotations
from typing import Tuple, Optional
import torch
from game3d.board.board import Board


class OccupancyCache:
    __slots__ = ("mask",)

    def __init__(self, board: Board) -> None:
        self.rebuild(board)          # initial build

    # ----------------------------------------------------------
    # public queries – vectorised
    # ----------------------------------------------------------
    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """O(1) scalar query."""
        return bool(self.mask[z, y, x].item())

    def is_occupied_batch(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N,3) long tensor  (x,y,z)
        returns: (N,) bool tensor
        """
        return self.mask[coords[:, 2], coords[:, 1], coords[:, 0]]

    @property
    def count(self) -> int:
        """Total occupied squares."""
        return int(self.mask.sum().item())

    # ----------------------------------------------------------
    # rebuild – called by CacheManager after every make/undo
    # ----------------------------------------------------------
    def rebuild(self, board: Board) -> None:
        """Zero-copy view – shares memory with board planes."""
        self.mask = board.occupancy_mask()   # torch.bool (9,9,9)


