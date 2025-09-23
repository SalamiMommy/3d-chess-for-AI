"""Incremental cache for Geomancy blocked squares (5-ply expiry)."""

from __future__ import annotations
from typing import Dict, Tuple, Optional
from pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.geomancy import block_candidates
from game.move import Move


class GeomancyCache:
    __slots__ = ("_blocks", "_board")  # _blocks: square -> expiry_ply

    def __init__(self, board: Board) -> None:
        self._board = board
        self._blocks: Dict[Tuple[int, int, int], int] = {}  # sq -> ply_when_expires
        # initial state is empty (blocks added via submit_block)

    # ---------- public ----------
    def is_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        """True if square is blocked **now**."""
        expiry = self._blocks.get(sq, 0)
        if expiry == 0:
            return False
        if current_ply >= expiry:
            # auto-purge on read (lazy)
            del self._blocks[sq]
            return False
        return True

    def block_square(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        """
        Controller requests to block an unoccupied square.
        Returns False if square is occupied or already blocked.
        """
        if self._board.piece_at(sq) is not None:
            return False  # must be unoccupied
        if self.is_blocked(sq, current_ply):
            return False  # already blocked
        self._blocks[sq] = current_ply + 5  # 5 plies
        return True

    def apply_move(self, mv: Move, mover: Color, current_ply: int) -> None:
        self._board.apply_move(mv)
        self._purge_expired(current_ply)   # clean up old blocks

    def undo_move(self, mv: Move, mover: Color, current_ply: int) -> None:
        self._board.undo_move(mv)
        self._purge_expired(current_ply)

    # ---------- internals ----------
    def _purge_expired(self, current_ply: int) -> None:
        expired = [sq for sq, expiry in self._blocks.items() if current_ply >= expiry]
        for sq in expired:
            del self._blocks[sq]


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_geom_cache: Optional[GeomancyCache] = None


def init_geomancy_cache(board: Board) -> None:
    global _geom_cache
    _geom_cache = GeomancyCache(board)


def get_geomancy_cache() -> GeomancyCache:
    if _geom_cache is None:
        raise RuntimeError("GeomancyCache not initialised")
    return _geom_cache
