"""Incremental cache for Armour piece immunity to pawn captures."""

from __future__ import annotations
from typing import Dict, Set, Tuple
from game3d.pieces.enums import Color, PieceType
from game3d.board.board import Board
from game3d.movement.movepiece import Move

class ArmourCache:
    __slots__ = ("_armoured_squares",)

    def __init__(self) -> None:
        # Set of squares containing Armour pieces (immune to pawn capture)
        self._armoured_squares: Set[Tuple[int, int, int]] = set()

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
        """Rebuild set of armoured squares."""
        self._armoured_squares.clear()
        for coord, piece in board.list_occupied():
            if piece.ptype == PieceType.ARMOUR:
                self._armoured_squares.add(coord)
