"""Incremental cache for Armoured ‘behind’ squares per WALL."""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
from pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.capturefrombehind import from_behind_squares
from game.move import Move


class ArmouredCache:
    __slots__ = ("_behind", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        self._behind: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        self._rebuild()

    # ---------- public ----------
    def can_capture(self, attacker_sq: Tuple[int, int, int], wall_sq: Tuple[int, int, int], controller: Color) -> bool:
        """True if attacker is ‘behind’ the wall (or wall not armoured)."""
        behind = self._behind.get(wall_sq)
        if behind is None:
            return True  # not a wall or no cache entry
        return attacker_sq in behind

    def apply_move(self, mv: Move, mover: Color) -> None:
        self._board.apply_move(mv)
        self._rebuild()   # walls may have moved

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._rebuild()

    # ---------- internals ----------
    def _rebuild(self) -> None:
        self._behind = from_behind_squares(self._board, Color.WHITE) | from_behind_squares(self._board, Color.BLACK)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_arm_cache: Optional[ArmouredCache] = None


def init_armoured_cache(board: Board) -> None:
    global _arm_cache
    _arm_cache = ArmouredCache(board)


def get_armoured_cache() -> ArmouredCache:
    if _arm_cache is None:
        raise RuntimeError("ArmouredCache not initialised")
    return _arm_cache
