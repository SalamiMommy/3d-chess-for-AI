"""Incremental cache for Archery attack targets."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.archery import archery_targets
from game3d.movement.movepiece import Move

class ArcheryCache:
    __slots__ = ("_targets",)

    def __init__(self) -> None:
        self._targets: Dict[Color, List[Tuple[int, int, int]]] = {
            Color.WHITE: [],
            Color.BLACK: [],
        }
        # no _board anymore

    def attack_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        return self._targets[controller]

    def is_valid_attack(self, sq: Tuple[int, int, int], controller: Color) -> bool:
        return sq in self._targets[controller]

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        # Only rebuild if the move affects relevant pieces
        if not self._move_affects_cache(mv, board):
            return
        self._rebuild(board)

    def _move_affects_cache(self, mv: Move, board: Board) -> bool:
        # Check if move involves pieces that affect this cache
        # For archery: check if archers or potential targets moved
        from_piece = board.piece_at(mv.from_coord)
        to_piece = board.piece_at(mv.to_coord)
        relevant_types = {PieceType.ARCHER, PieceType.KING, PieceType.PRIEST}  # etc.
        return (from_piece and from_piece.ptype in relevant_types) or \
            (to_piece and to_piece.ptype in relevant_types)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Board is already guaranteed to be current – just rebuild."""
        self._rebuild(board)

    def _rebuild(self, board: Board) -> None:
        for col in (Color.WHITE, Color.BLACK):
            self._targets[col] = archery_targets(board, col)


