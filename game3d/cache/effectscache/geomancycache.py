# geomancycache.py (updated with common modules)
from __future__ import annotations
from typing import Dict, Tuple, Optional, TYPE_CHECKING

from game3d.common.enums import Color
from game3d.common.coord_utils import Coord
from game3d.common.cache_utils import get_piece
from game3d.pieces.pieces.geomancer import block_candidates
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.board.board import Board

class GeomancyCache:
    __slots__ = ("_blocks", "_cache_manager")

    def __init__(self, cache_manager=None) -> None:
        self._blocks: Dict[Coord, int] = {}
        self._cache_manager = cache_manager

    def is_blocked(self, sq: Coord, current_ply: int) -> bool:
        expiry = self._blocks.get(sq, 0)
        if expiry == 0 or current_ply >= expiry:
            self._blocks.pop(sq, None)
            return False
        return True

    def block_square(self, sq: Coord, current_ply: int, board: Board) -> bool:
        if self._cache_manager and get_piece(self._cache_manager, sq) is not None:
            return False
        if self.is_blocked(sq, current_ply):
            return False
        self._blocks[sq] = current_ply + 5
        return True

    def apply_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        # FIXED: Use cache_manager instead of board
        candidates = block_candidates(self._cache_manager, mover)

        for sq in candidates:
            if not self.is_blocked(sq, current_ply):
                self._blocks[sq] = current_ply + 5

        if current_ply % 10 == 0:
            self._cleanup_expired(current_ply)

    def undo_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        if current_ply % 10 == 0:
            self._cleanup_expired(current_ply)

    def _cleanup_expired(self, current_ply: int) -> None:
        expired_squares = [sq for sq, expiry in self._blocks.items() if current_ply >= expiry]
        for sq in expired_squares:
            del self._blocks[sq]

    def clear(self) -> None:
        self._blocks.clear()

    def get_stats(self) -> Dict[str, int]:
        return {
            'blocked_squares': len(self._blocks),
        }

    def set_cache_manager(self, cache_manager: 'OptimizedCacheManager') -> None:
        """Set the cache manager reference - ensures single instance usage"""
        self._cache_manager = cache_manager
