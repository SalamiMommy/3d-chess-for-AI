"""Incremental cache for Movement-Debuff squares."""

from __future__ import annotations
from typing import Set, Tuple, Optional, Dict
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.movementdebuff import debuffed_squares
from game3d.movement.movepiece import Move

class MovementDebuffCache:
    __slots__ = ("_debuffed",)

    def __init__(self) -> None:
        self._debuffed: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

    # ---------- public ----------
    def is_debuffed(self, sq: Tuple[int, int, int], victim_color: Color) -> bool:
        return sq in self._debuffed[victim_color]

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    # ---------- internals ----------
    def _rebuild(self, board: Board) -> None:
        self._debuffed[Color.WHITE] = debuffed_squares(board, Color.BLACK)
        self._debuffed[Color.BLACK] = debuffed_squares(board, Color.WHITE)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_debuff_cache: Optional[MovementDebuffCache] = None

def init_movement_debuff_cache() -> None:
    global _debuff_cache
    _debuff_cache = MovementDebuffCache()

def get_movement_debuff_cache() -> MovementDebuffCache:
    if _debuff_cache is None:
        raise RuntimeError("MovementDebuffCache not initialised")
    return _debuff_cache
