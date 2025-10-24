# trailblazecache.py (updated with common modules)
from __future__ import annotations
from typing import Dict, Set, TYPE_CHECKING

from game3d.common.coord_utils import Coord
from game3d.common.enums import Color, PieceType
from game3d.common.piece_utils import iterate_occupied, get_pieces_by_type
from game3d.common.cache_utils import get_piece
from game3d.movement.movepiece import Move
from game3d.pieces.pieces.trailblazer import TrailblazeRecorder

if TYPE_CHECKING:
    from game3d.board.board import Board

class TrailblazeCache:
    def __init__(self, cache_manager) -> None:
        self._recorders: Dict[Coord, TrailblazeRecorder] = {}
        self._counters: Dict[Coord, int] = {}
        self._cache_manager = cache_manager

    def current_trail_squares(self, controller: Color, board: Board) -> Set[Coord]:
        result: Set[Coord] = set()
        for coord, recorder in self._recorders.items():
            piece = get_piece(self._cache_manager, coord)
            if piece and piece.color == controller and piece.ptype == PieceType.TRAILBLAZER:
                result.update(recorder.current_trail())
        return result

    def mark_trail(self, trailblazer_pos: Coord, path: Set[Coord]) -> None:
        if trailblazer_pos not in self._recorders:
            self._recorders[trailblazer_pos] = TrailblazeRecorder()
        self._recorders[trailblazer_pos].add_trail(path)

    def record_trail(self, trailblazer_pos: Coord, path: Set[Coord]) -> None:
        self.mark_trail(trailblazer_pos, path)

    def increment_counter(self, sq: Coord, enemy_color: Color, board: Board) -> bool:
        if sq not in self.current_trail_squares(enemy_color, board):
            return False
        self._counters[sq] = self._counters.get(sq, 0) + 1
        return self._counters[sq] >= 3

    def apply_move(self, mv: Move, mover: Color, current_ply: int, board: Board) -> None:
        self._sync_recorders_with_board(board)

    def _sync_recorders_with_board(self, board: Board) -> None:
        existing_coords = {coord for coord, piece in iterate_occupied(board) if piece.ptype == PieceType.TRAILBLAZER}
        self._recorders = {
            coord: recorder
            for coord, recorder in self._recorders.items()
            if coord in existing_coords
        }

    def clear(self) -> None:
        self._recorders.clear()
        self._counters.clear()
