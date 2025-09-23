"""Incremental cache for Movement-Buff squares."""

from __future__ import annotations
from typing import Set, Tuple, Optional
from pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.movementbuff import buffed_squares


class MovementBuffCache:
    __slots__ = ("_buffed", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        self._buffed: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }
        self._rebuild()

    # ---------- public ----------
    def is_buffed(self, sq: Tuple[int, int, int], friendly_color: Color) -> bool:
        return sq in self._buffed[friendly_color]

    def apply_move(self, mv: Move, mover: Color) -> None:
        self._board.apply_move(mv)
        self._rebuild()

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._rebuild()

    # ---------- internals ----------
    def _rebuild(self) -> None:
        self._buffed[Color.WHITE] = buffed_squares(self._board, Color.WHITE)
        self._buffed[Color.BLACK] = buffed_squares(self._board, Color.BLACK)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_buff_cache: Optional[MovementBuffCache] = None


def init_movement_buff_cache(board: Board) -> None:
    global _buff_cache
    _buff_cache = MovementBuffCache(board)


def get_movement_buff_cache() -> MovementBuffCache:
    if _buff_cache is None:
        raise RuntimeError("MovementBuffCache not initialised")
    return _buff_cache
