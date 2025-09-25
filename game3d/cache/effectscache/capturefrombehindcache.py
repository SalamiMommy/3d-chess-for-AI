"""Incremental cache for Armoured ‘behind’ squares per WALL."""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.capturefrombehind import from_behind_squares
from game3d.movement.movepiece import Move

class ArmouredCache:
    __slots__ = ("_behind",)

    def __init__(self) -> None:
        self._behind: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}

    # ---------- public ----------
    def can_capture(self, attacker_sq: Tuple[int, int, int], wall_sq: Tuple[int, int, int], controller: Color) -> bool:
        behind = self._behind.get(wall_sq)
        if behind is None:
            return True
        return attacker_sq in behind

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    # ---------- internals ----------
    def _rebuild(self, board: Board) -> None:
        self._behind = (from_behind_squares(board, Color.WHITE) |
                        from_behind_squares(board, Color.BLACK))


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_arm_cache: Optional[ArmouredCache] = None

def init_armoured_cache() -> None:
    global _arm_cache
    _arm_cache = ArmouredCache()

def get_armoured_cache() -> ArmouredCache:
    if _arm_cache is None:
        raise RuntimeError("ArmouredCache not initialised")
    return _arm_cache
