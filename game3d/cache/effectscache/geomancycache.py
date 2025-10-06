"""Incremental cache for Geomancy blocked squares (5-ply expiry)."""

from __future__ import annotations
from typing import Dict, Tuple
from game3d.pieces.enums import Color
from game3d.effects.geomancy import block_candidates
from game3d.movement.movepiece import Move

class GeomancyCache:
    __slots__ = ("_blocks",)

    def __init__(self) -> None:
        self._blocks: Dict[Tuple[int, int, int], int] = {}

    def is_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        """Check if square is blocked and clean up expired entries."""
        expiry = self._blocks.get(sq, 0)
        if expiry == 0 or current_ply >= expiry:
            self._blocks.pop(sq, None)
            return False
        return True

    def block_square(self, sq: Tuple[int, int, int], current_ply: int, board: Board) -> bool:
        """Manually block a square if empty and not already blocked."""
        if cache.piece_cache.get(sq) is not None:
            return False
        if self.is_blocked(sq, current_ply):
            return False
        self._blocks[sq] = current_ply + 5
        return True

    def apply_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        """Apply geomancy blocking for the player who just moved."""
        # ← FIX: Use mover, not mover.opposite()
        for sq in block_candidates(board, mover):
            if not self.is_blocked(sq, current_ply):
                self._blocks[sq] = current_ply + 5

        # Optional: Clean up some expired blocks to prevent memory growth
        # (Only if performance allows - this is O(n))
        if current_ply % 10 == 0:  # Every 10 plies
            self._cleanup_expired(current_ply)

    def undo_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        """Optional: Clean up expired blocks."""
        # For simplicity, we rely on is_blocked() to clean on access
        # But periodic cleanup helps with memory
        if current_ply % 10 == 0:
            self._cleanup_expired(current_ply)

    def _cleanup_expired(self, current_ply: int) -> None:
        """Remove all expired blocks to prevent memory leaks."""
        expired_squares = [sq for sq, expiry in self._blocks.items() if current_ply >= expiry]
        for sq in expired_squares:
            del self._blocks[sq]
