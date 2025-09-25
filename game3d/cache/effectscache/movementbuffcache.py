"""Incremental cache for Movement-Buff squares."""

from __future__ import annotations
from typing import Set, Tuple, Optional, Dict
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.movementbuff import buffed_squares
from game3d.movement.movepiece import Move

class MovementBuffCache:
    __slots__ = ("_buffed",)

    def __init__(self) -> None:
        self._buffed: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

    # ---------- public ----------
    def is_buffed(self, sq: Tuple[int, int, int], friendly_color: Color) -> bool:
        return sq in self._buffed[friendly_color]

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    # ---------- internals ----------
    def _rebuild(self, board: Board) -> None:
        self._buffed[Color.WHITE] = buffed_squares(board, Color.WHITE)
        self._buffed[Color.BLACK] = buffed_squares(board, Color.BLACK)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_buff_cache: Optional[MovementBuffCache] = None

def init_movement_buff_cache() -> None:
    global _buff_cache
    _buff_cache = MovementBuffCache()

def get_movement_buff_cache() -> MovementBuffCache:
    if _buff_cache is None:
        raise RuntimeError("MovementBuffCache not initialised")
    return _buff_cache
