"""Incremental cache for Armour piece immunity to pawn captures."""

from __future__ import annotations
from typing import Dict, Set, Tuple, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.board.board import Board

class ArmourCache:
    __slots__ = ("_armoured_squares", "_cache_manager")

    def __init__(self, cache_manager=None) -> None:
        # Set of squares containing Armour pieces (immune to pawn capture)
        self._armoured_squares: Set[Tuple[int, int, int]] = set()
        # Cache manager reference for consistency with other caches
        self._cache_manager = cache_manager

    def is_armoured(self, sq: Tuple[int, int, int]) -> bool:
        """Check if square contains an Armour piece (immune to pawn capture)."""
        return sq in self._armoured_squares

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Rebuild armour cache after move."""
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Rebuild armour cache after undo."""
        self._rebuild(board)

    def _rebuild(self, board: Board) -> None:
        """Rebuild set of armoured squares using cache manager's piece cache."""
        self._armoured_squares.clear()

        # Use cache manager's piece cache if available
        if self._cache_manager:
            # Iterate over all board coordinates
            for x in range(9):
                for y in range(9):
                    for z in range(9):
                        coord = (x, y, z)
                        piece = self._cache_manager.piece_cache.get(coord)
                        if piece and piece.ptype == PieceType.ARMOUR:
                            self._armoured_squares.add(coord)
        else:
            # Fallback to board method if cache manager not available
            for coord, piece in board.list_occupied():
                if piece.ptype == PieceType.ARMOUR:
                    self._armoured_squares.add(coord)

    def clear(self) -> None:
        """Clear all cached data."""
        self._armoured_squares.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics for performance monitoring."""
        return {
            'armoured_squares': len(self._armoured_squares),
        }
    def can_capture(self, attacker_sq, wall_sq, controller) -> bool:
        """
        Armour never helps anyone *capture* a wall – it only *blocks* pawn
        captures against itself.  Hence this cache always returns False.
        """
        return False
