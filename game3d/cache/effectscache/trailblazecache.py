from __future__ import annotations
from typing import Dict, Set
from game3d.common.common import Coord  # Assuming Coord is Tuple[int, int, int]
from game3d.common.protocols import BoardProto  # If needed for type checking
from game3d.effects.trailblazing import TrailblazeRecorder
from game3d.pieces.enums import Color, PieceType
from game3d.board.board import Board
from game3d.movement.movepiece import Move

class TrailblazeCache:
    def __init__(self) -> None:
        self._recorders: Dict[Coord, TrailblazeRecorder] = {}
        self._counters: Dict[Coord, int] = {}

    def record_trail(self, trailblazer_pos: Coord, path: Set[Coord]) -> None:
        """Record a new trail for the trailblazer at trailblazer_pos."""
        if trailblazer_pos not in self._recorders:
            self._recorders[trailblazer_pos] = TrailblazeRecorder()
        self._recorders[trailblazer_pos].add_trail(path)

    def current_trail_squares(self, controller: Color, board: Board) -> Set[Coord]:
        """Get all squares in trails of friendly trailblazers."""
        result: Set[Coord] = set()
        for coord, recorder in self._recorders.items():
            piece = cache.piece_cache.get(coord)
            if piece and piece.color == controller and piece.ptype == PieceType.TRAILBLAZER:
                result.update(recorder.current_trail())
        return result

    def increment_counter(self, sq: Coord, enemy_color: Color, board: Board) -> bool:
        """Check if square is in enemy trail and increment counter."""
        if sq not in self.current_trail_squares(enemy_color.opposite(), board):
            return False
        self._counters[sq] = self._counters.get(sq, 0) + 1
        return self._counters[sq] >= 3

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Handle trailblazer creation/destruction (promotion, capture, etc.)."""
        self._sync_recorders_with_board(board)

    def _sync_recorders_with_board(self, board: Board) -> None:
        """Remove recorders for trailblazers that no longer exist."""
        existing_coords = {coord for coord, piece in board.list_occupied() if piece.ptype == PieceType.TRAILBLAZER}

        # Keep only recorders for trailblazers that still exist
        self._recorders = {
            coord: recorder
            for coord, recorder in self._recorders.items()
            if coord in existing_coords
        }
