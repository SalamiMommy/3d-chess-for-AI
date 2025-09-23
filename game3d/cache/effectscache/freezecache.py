"""Incremental cache for frozen enemy squares."""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
from pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.freeze import frozen_squares


class FreezeCache:
    __slots__ = ("_frozen", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        self._frozen: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }
        self._rebuild()

    # ----------------------------------------------------------
    # public
    # ----------------------------------------------------------
    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        x, y, z = sq
        if not self.occupancy[z, y, x].item():
            return False

    def apply_move(self, mv: Move, mover: Color) -> None:
        """Update cache after move (full rebuild for now – O(V) but V=729)."""
        self._board.apply_move(mv)
        self._rebuild()

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._rebuild()

    # ----------------------------------------------------------
    # internals
    # ----------------------------------------------------------
    def _rebuild(self) -> None:
        self._frozen[Color.WHITE] = frozen_squares(self._board, Color.BLACK)
        self._frozen[Color.BLACK] = frozen_squares(self._board, Color.WHITE)


# ------------------------------------------------------------------
# module-level singleton
# ------------------------------------------------------------------
_freeze_cache: Optional[FreezeCache] = None


def init_freeze_cache(board: Board) -> None:
    global _freeze_cache
    _freeze_cache = FreezeCache(board)


def get_freeze_cache() -> FreezeCache:
    if _freeze_cache is None:
        raise RuntimeError("FreezeCache not initialised")
    return _freeze_cache
