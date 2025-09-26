from __future__ import annotations
from typing import Dict, Tuple, Set
from game3d.pieces.enums import Color, PieceType
from game3d.board.board import Board
from game3d.effects.trailblazing import squares_to_mark, TrailblazeRecorder
from game3d.movement.movepiece import Move

class TrailblazeCache:
    __slots__ = ("_counters", "_recorders")

    def __init__(self) -> None:
        self._counters: Dict[Tuple[int, int, int], int] = {}  # square -> 0..3
        self._recorders: Dict[Tuple[int, int, int], TrailblazeRecorder] = {}

    def mark_trail(self, trailblazer_sq: Tuple[int, int, int], slid_squares: Set[Tuple[int, int, int]]) -> None:
        rec = self._recorders.get(trailblazer_sq)
        if rec is not None:
            rec.add_trail(slid_squares)
            # Do NOT call _refresh_counter_keys — we can't access board here.
            # Stale counters are harmless; they're validated at use time.

    def current_trail_squares(self, controller: Color, board: Board) -> Set[Tuple[int, int, int]]:
        out: Set[Tuple[int, int, int]] = set()
        for coord, rec in self._recorders.items():
            piece = board.piece_at(coord)
            if piece is not None and piece.color == controller and piece.ptype == PieceType.TRAILBLAZER:
                out.update(rec.current_trail())
        return out

    def increment_counter(self, sq: Tuple[int, int, int], enemy_color: Color, board: Board) -> bool:
        # Only count if square is currently in enemy's trail
        if sq not in self.current_trail_squares(enemy_color.opposite(), board):
            return False
        self._counters[sq] = self._counters.get(sq, 0) + 1
        return self._counters[sq] >= 3

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild_recorders(board)
        # Optional: clear counters on full rebuild to save memory
        # self._counters.clear()

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild_recorders(board)
        # self._counters.clear()

    def _rebuild_recorders(self, board: Board) -> None:
        white = squares_to_mark(board, Color.WHITE)
        black = squares_to_mark(board, Color.BLACK)
        self._recorders = {**white, **black}
