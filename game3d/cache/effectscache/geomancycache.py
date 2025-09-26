from __future__ import annotations
"""Incremental cache for Geomancy blocked squares (5-ply expiry)."""
#game3d/cache/effects/geomancycache.py
"""Incremental cache for Geomancy blocked squares (5-ply expiry)."""


from typing import Dict, Tuple
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.geomancy import block_candidates
from game3d.movement.movepiece import Move

class GeomancyCache:
    __slots__ = ("_blocks",)

    def __init__(self) -> None:
        self._blocks: Dict[Tuple[int, int, int], int] = {}

    def is_blocked(self, sq: Tuple[int, int, int], current_ply: int) -> bool:
        expiry = self._blocks.get(sq, 0)
        if expiry == 0 or current_ply >= expiry:
            self._blocks.pop(sq, None)
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
        current_controller = mover.opposite()
        for sq in block_candidates(board, current_controller):
            if not self.is_blocked(sq, current_ply):
                self._blocks[sq] = current_ply + 5

    def undo_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        # Optional: purge blocks with expiry > current_ply
        # For simplicity, do nothing — expired blocks cleaned on access
        pass
