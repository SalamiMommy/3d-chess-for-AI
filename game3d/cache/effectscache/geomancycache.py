"""Incremental cache for Geomancy blocked squares (5-ply expiry)."""

from __future__ import annotations
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from game3d.common.enums import Color
from game3d.pieces.pieces.geomancer import block_candidates
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.board.board import Board

class GeomancyCache:
    __slots__ = ("_blocks", "_cache_manager")

    def __init__(self, cache_manager=None) -> None:
        self._blocks: Dict[Tuple[int, int, int], int] = {}
        # Cache manager reference
        self._cache_manager = cache_manager

    def is_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        """Check if square is blocked and clean up expired entries."""
        expiry = self._blocks.get(sq, 0)
        if expiry == 0 or current_ply >= expiry:
            self._blocks.pop(sq, None)
            return False
        return True

    def block_square(self, sq: Tuple[int, int, int], current_ply: int, board: Board) -> bool:
        """Manually block a square if empty and not already blocked."""
        # Use cache manager to check if square is occupied
        if self._cache_manager:
            piece = self._cache_manager.occupancy.get(sq)
        else:
            piece = None  # Conservative: assume occupied if no cache

        if piece is not None:
            return False
        if self.is_blocked(sq, current_ply):
            return False
        self._blocks[sq] = current_ply + 5
        return True

    def apply_move(self, mv: Move, mover: Color, current_ply: int, board: "Board") -> None:
        """Apply geomancy blocking for the player who just moved."""
        # 1.  Keep occupancy in sync *before* we touch geomancy caches
        captured_sq = mv.to_coord if mv.is_capture else None
        self._update_occupancy_incrementally(board, mv.from_coord, captured_sq)

        # 2.  Classic geomancy-cache logic (unchanged)
        if self._cache_manager:
            candidates = block_candidates(board, mover, self._cache_manager)
        else:
            candidates = block_candidates(board, mover, None)

        for sq in candidates:
            if not self.is_blocked(sq, current_ply):
                self._blocks[sq] = current_ply + 5

        # Optional: Clean up some expired blocks to prevent memory growth
        if current_ply % 10 == 0:
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

    def clear(self) -> None:
        """Clear all cached data."""
        self._blocks.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics for performance monitoring."""
        return {
            'blocked_squares': len(self._blocks),
        }

    def _update_occupancy_incrementally(
        self,
        board: "Board",
        moved_sq: Coord,
        captured_sq: Optional[Coord],
    ) -> None:
        """
        Mirror the board state into the occupancy cache without a full
        rebuild.  Only the squares that *really* changed are written.
        """
        occ = self._cache_manager.occupancy  # type: OccupancyCache

        # 1.  Old piece left the from-square
        occ.set_position(moved_sq, None)

        # 2.  Captured piece disappeared (if any)
        if captured_sq is not None:
            occ.set_position(captured_sq, None)

        # 3.  Moved piece arrived at the to-square
        to_piece = board.get(moved_sq)  # board already reflects the move
        occ.set_position(moved_sq, to_piece)
