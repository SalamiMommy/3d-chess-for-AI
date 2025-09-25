"""Incremental cache for Geomancy blocked squares (5-ply expiry)."""

from __future__ import annotations
from typing import Dict, Tuple, Optional
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.geomancy import block_candidates
from game3d.movement.movepiece import Move

class GeomancyCache:
    __slots__ = ("_blocks",)

    def __init__(self) -> None:
        self._blocks: Dict[Tuple[int, int, int], int] = {}  # sq -> expiry_ply

    # ---------- public ----------
    def is_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        expiry = self._blocks.get(sq, 0)
        if expiry == 0:
            return False
        if current_ply >= expiry:
            del self._blocks[sq]
            return False
        return True

    def block_square(self, sq: Tuple[int, int, int], current_ply: int, board: Board) -> bool:
        if board.piece_at(sq) is not None:
            return False
        if self.is_blocked(sq, current_ply):
            return False
        self._blocks[sq] = current_ply + 5
        return True

    def apply_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        self._purge_expired(current_ply)

    def undo_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        self._purge_expired(current_ply)

    # ---------- internals ----------
    def _purge_expired(self, current_ply: int) -> None:
        for sq in [sq for sq, ex in self._blocks.items() if current_ply >= ex]:
            del self._blocks[sq]


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_geom_cache: Optional[GeomancyCache] = None

def init_geomancy_cache() -> None:
    global _geom_cache
    _geom_cache = GeomancyCache()

def get_geomancy_cache() -> GeomancyCache:
    if _geom_cache is None:
        raise RuntimeError("GeomancyCache not initialised")
    return _geom_cache
