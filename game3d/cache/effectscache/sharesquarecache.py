"""Multi-occupancy cache for Share-Square (Knights only)."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.board.board import Board

class ShareSquareCache:
    __slots__ = ("_stack", "_cache_manager")

    def __init__(self, cache_manager=None) -> None:
        self._stack: Dict[Tuple[int, int, int], List[Piece]] = {}
        # Cache manager reference for consistency with other caches
        self._cache_manager = cache_manager

    def pieces_at(self, sq: Tuple[int, int, int]) -> List[Piece]:
        return self._stack.get(sq, [])

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional[Piece]:
        stack = self._stack.get(sq)
        return stack[-1] if stack else None

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Rebuild cache from current board state."""
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Rebuild cache from current board state."""
        self._rebuild(board)

    def _rebuild(self, board: Board) -> None:
        """Rebuild entire cache from board using cache manager's piece cache."""
        self._stack.clear()

        # Use cache manager's piece cache if available
        if self._cache_manager:
            # Iterate over all board coordinates
            for x in range(9):
                for y in range(9):
                    for z in range(9):
                        coord = (x, y, z)
                        piece = self._cache_manager.piece_cache.get(coord)
                        if piece and piece.ptype == PieceType.KNIGHT:
                            self._stack.setdefault(coord, []).append(piece)
        else:
            # Fallback to board method if cache manager not available
            for coord, piece in board.list_occupied():
                if piece.ptype == PieceType.KNIGHT:
                    self._stack.setdefault(coord, []).append(piece)

    def clear(self) -> None:
        """Clear all cached data."""
        self._stack.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics for performance monitoring."""
        total_knights = sum(len(stack) for stack in self._stack.values())
        occupied_squares = len(self._stack)
        return {
            'total_knights': total_knights,
            'occupied_squares': occupied_squares,
        }
