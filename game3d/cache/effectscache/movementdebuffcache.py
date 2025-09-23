"""Incremental cache for Movement-Debuff squares."""

from __future__ import annotations
from typing import Set, Tuple, Optional, Dict
from pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.movementdebuff import debuffed_squares


class MovementDebuffCache:
    __slots__ = ("_debuffed", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        self._debuffed: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }
        self._rebuild()

    # ---------- public ----------
    def is_debuffed(self, sq: Tuple[int, int, int], victim_color: Color) -> bool:
        return sq in self._debuffed[victim_color]

    def apply_move(self, mv: Move, mover: Color) -> None:
        self._board.apply_move(mv)
        self._rebuild()

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._rebuild()

    # ---------- internals ----------
    def _rebuild(self) -> None:
        self._debuffed[Color.WHITE] = debuffed_squares(self._board, Color.BLACK)
        self._debuffed[Color.BLACK] = debuffed_squares(self._board, Color.WHITE)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_debuff_cache: Optional[MovementDebuffCache] = None


def init_movement_debuff_cache(board: Board) -> None:
    global _debuff_cache
    _debuff_cache = MovementDebuffCache(board)


def get_movement_debuff_cache() -> MovementDebuffCache:
    if _debuff_cache is None:
        raise RuntimeError("MovementDebuffCache not initialised")
    return _debuff_cache
