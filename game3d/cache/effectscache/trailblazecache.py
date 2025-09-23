"""Incremental cache for Trailblazing counters."""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Set
from pieces.enums import Color, PieceType
from game3d.board.board import Board
from game3d.effects.auras.trailblazing import squares_to_mark, TrailblazeRecorder
from game.move import Move


class TrailblazeCache:
    __slots__ = ("_counters", "_recorders", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        self._counters: Dict[Tuple[int, int, int], int] = {}  # square -> 0..3
        self._recorders: Dict[Tuple[int, int, int], TrailblazeRecorder] = {}
        self._rebuild_recorders()

    # ---------- public ----------
    def mark_trail(self, trailblazer_sq: Tuple[int, int, int], slid_squares: Set[Tuple[int, int, int]]) -> None:
        """Call from slider generator – stores the trail."""
        rec = self._recorders.get(trailblazer_sq)
        if rec is not None:
            rec.add_trail(slid_squares)
            # refresh counter keys (old squares may expire)
            self._refresh_counter_keys()

    def current_trail_squares(self, controller: Color) -> Set[Tuple[int, int, int]]:
        """All squares currently blazing for controller."""
        out: Set[Tuple[int, int, int]] = set()
        for coord, rec in self._recorders.items():
            piece = self._board.piece_at(coord)
            if piece is not None and piece.color == controller and piece.ptype == PieceType.TRAILBLAZER:
                out.update(rec.current_trail())
        return out

    def increment_counter(self, sq: Tuple[int, int, int], enemy_color: Color) -> bool:
        """
        Enemy piece ended move / step on sq.
        Returns True → counter hit 3 → caller must remove piece (unless king + priests).
        """
        if sq not in self.current_trail_squares(enemy_color.opposite()):
            return False
        self._counters[sq] = self._counters.get(sq, 0) + 1
        return self._counters[sq] >= 3

    def apply_move(self, mv: Move, mover: Color) -> None:
        self._board.apply_move(mv)
        self._rebuild_recorders()   # trailblazers may have moved

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._rebuild_recorders()

    # ---------- internals ----------
    def _rebuild_recorders(self) -> None:
        self._recorders = squares_to_mark(self._board, Color.WHITE) | squares_to_mark(self._board, Color.BLACK)

    def _refresh_counter_keys(self) -> None:
        """Purge counters whose square is no longer in any trail."""
        current = self.current_trail_squares(Color.WHITE) | self.current_trail_squares(Color.BLACK)
        to_del = [sq for sq in self._counters if sq not in current]
        for sq in to_del:
            del self._counters[sq]


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_trail_cache: Optional[TrailblazeCache] = None


def init_trailblaze_cache(board: Board) -> None:
    global _trail_cache
    _trail_cache = TrailblazeCache(board)


def get_trailblaze_cache() -> TrailblazeCache:
    if _trail_cache is None:
        raise RuntimeError("TrailblazeCache not initialised")
    return _trail_cache
