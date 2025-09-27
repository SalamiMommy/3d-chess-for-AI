"""Multi-occupancy cache for Share-Square (Knights only)."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from game3d.pieces.enums import Color, PieceType
from game3d.board.board import Board
from game3d.pieces.piece import Piece
from game3d.movement.movepiece import Move

class ShareSquareCache:
    __slots__ = ("_stack",)

    def __init__(self) -> None:
        self._stack: Dict[Tuple[int, int, int], List[Piece]] = {}

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
        """Rebuild entire cache from board."""
        self._stack.clear()
        for coord, piece in board.list_occupied():
            if piece.ptype == PieceType.KNIGHT:
                self._stack.setdefault(coord, []).append(piece)

    # Remove add_knight/remove_knight - they're not needed with rebuild approach
