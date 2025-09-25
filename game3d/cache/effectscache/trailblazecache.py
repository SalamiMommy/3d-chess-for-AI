"""Incremental cache for Trailblazing counters."""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Set
from game3d.pieces.enums import Color, PieceType
from game3d.board.board import Board
from game3d.effects.trailblazing import squares_to_mark, TrailblazeRecorder
from game3d.movement.movepiece import Move

class TrailblazeCache:
    __slots__ = ("_counters", "_recorders")

    def __init__(self) -> None:
        self._counters: Dict[Tuple[int, int, int], int] = {}  # square -> 0..3
        self._recorders: Dict[Tuple[int, int, int], TrailblazeRecorder] = {}

    # ---------- public ----------
    def mark_trail(self, trailblazer_sq: Tuple[int, int, int], slid_squares: Set[Tuple[int, int, int]]) -> None:
        rec = self._recorders.get(trailblazer_sq)
        if rec is not None:
            rec.add_trail(slid_squares)
            self._refresh_counter_keys()

    def current_trail_squares(self, controller: Color, board: Board) -> Set[Tuple[int, int, int]]:
        out: Set[Tuple[int, int, int]] = set()
        for coord, rec in self._recorders.items():
            piece = board.piece_at(coord)
            if piece is not None and piece.color == controller and piece.ptype == PieceType.TRAILBLAZER:
                out.update(rec.current_trail())
        return out

    def increment_counter(self, sq: Tuple[int, int, int], enemy_color: Color, board: Board) -> bool:
        if sq not in self.current_trail_squares(enemy_color.opposite(), board):
            return False
        self._counters[sq] = self._counters.get(sq, 0) + 1
        return self._counters[sq] >= 3

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild_recorders(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild_recorders(board)

    # ---------- internals ----------
    def _rebuild_recorders(self, board: Board) -> None:
        self._recorders = squares_to_mark(board, Color.WHITE) | squares_to_mark(board, Color.BLACK)

    def _refresh_counter_keys(self) -> None:
        current = (self.current_trail_squares(Color.WHITE) |
                   self.current_trail_squares(Color.BLACK))
        for sq in [k for k in self._counters if k not in current]:
            del self._counters[sq]


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_trail_cache: Optional[TrailblazeCache] = None

def init_trailblaze_cache() -> None:
    global _trail_cache
    _trail_cache = TrailblazeCache()

def get_trailblaze_cache() -> TrailblazeCache:
    if _trail_cache is None:
        raise RuntimeError("TrailblazeCache not initialised")
    return _trail_cache
