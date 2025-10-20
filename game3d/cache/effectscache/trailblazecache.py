# game3d/cache/effects_cache/trailblazecache.py  # Adjusted path

from __future__ import annotations
from typing import Dict, Set, TYPE_CHECKING
from game3d.common.coord_utils import Coord
from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.pieces.pieces.trailblazer import TrailblazeRecorder
from game3d.cache.caches.occupancycache import OccupancyCache

if TYPE_CHECKING:
    from game3d.board.board import Board

class TrailblazeCache:
    def __init__(self, cache_manager) -> None:  # Assume always present
        self._recorders: Dict[Coord, TrailblazeRecorder] = {}
        self._counters: Dict[Coord, int] = {}
        self._cache_manager = cache_manager

    def current_trail_squares(self, controller: Color, board: Board) -> Set[Coord]:
        """Get all squares in trails of friendly trailblazers."""
        result: Set[Coord] = set()
        for coord, recorder in self._recorders.items():
            piece = self._cache_manager.occupancy.get(coord)
            if piece and piece.color == controller and piece.ptype == PieceType.TRAILBLAZER:
                result.update(recorder.current_trail())
        return result

    def mark_trail(self, trailblazer_pos: Coord, path: Set[Coord]) -> None:
        """Record a new trail for the trailblazer at trailblazer_pos."""
        if trailblazer_pos not in self._recorders:
            self._recorders[trailblazer_pos] = TrailblazeRecorder()
        self._recorders[trailblazer_pos].add_trail(path)

    def record_trail(self, trailblazer_pos: Coord, path: Set[Coord]) -> None:
        """Record a new trail for the trailblazer at trailblazer_pos."""
        self.mark_trail(trailblazer_pos, path)

    def increment_counter(self, sq: Coord, enemy_color: Color, board: Board) -> bool:
        """Check if square is in enemy trail and increment counter."""
        if sq not in self.current_trail_squares(enemy_color, board):  # Fixed: enemy_color for enemy trails
            return False
        self._counters[sq] = self._counters.get(sq, 0) + 1
        return self._counters[sq] >= 3

    def apply_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        """Handle trailblazer creation/destruction (promotion, capture, etc.)."""
        self._update_occupancy_incrementally(board, mv.from_coord, mv.to_coord)
        self._sync_recorders_with_board(board)

    def _sync_recorders_with_board(self, board: Board) -> None:
        """Remove recorders for trailblazers that no longer exist."""
        existing_coords = {coord for coord, piece in board.list_occupied() if piece.ptype == PieceType.TRAILBLAZER}
        self._recorders = {
            coord: recorder
            for coord, recorder in self._recorders.items()
            if coord in existing_coords
        }

    def clear(self) -> None:
        """Clear all cached trail data."""
        self._recorders.clear()
        self._counters.clear()

    def _update_occupancy_incrementally(
        self, board: "Board", from_sq: Coord, to_sq: Coord
    ) -> None:
        occ: OccupancyCache = self._cache_manager.occupancy
        self._cache_manager.set_piece(from_sq, None)
        # Cache-first read
        to_piece = self._cache_manager.occupancy.get(to_sq)
        self._cache_manager.set_piece(to_sq, to_piece)
