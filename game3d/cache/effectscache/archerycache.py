"""Incremental cache for Archery attack targets."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.archery import archery_targets
from game3d.movement.movepiece import Move

class ArcheryCache:
    __slots__ = ("_targets",)

    def __init__(self) -> None:
        self._targets: Dict[Color, List[Tuple[int, int, int]]] = {
            Color.WHITE: [],
            Color.BLACK: [],
        }
        # no _board anymore

    def attack_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        return self._targets[controller]

    def is_valid_attack(self, sq: Tuple[int, int, int], controller: Color) -> bool:
        return sq in self._targets[controller]

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Board is already guaranteed to be current – just rebuild."""
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Board is already guaranteed to be current – just rebuild."""
        self._rebuild(board)

    def _rebuild(self, board: Board) -> None:
        for col in (Color.WHITE, Color.BLACK):
            self._targets[col] = archery_targets(board, col)

_arch_cache: Optional[ArcheryCache] = None

def init_archery_cache() -> None:
    global _arch_cache
    _arch_cache = ArcheryCache()

def get_archery_cache() -> ArcheryCache:
    if _arch_cache is None:
        raise RuntimeError("ArcheryCache not initialised")
    return _arch_cache
