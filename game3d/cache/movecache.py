"""Incremental legal-move cache for 9×9×9."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pieces.enums import Color
from game.move import Move
from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
from game3d.movement.check import king_in_check
from game3d.board.board import Board

# global singleton
_move_cache: Optional[MoveCache] = None


class MoveCache:
    """Per-side legal-move cache with make/unmake."""

    __slots__ = ("_legal", "_board", "_current", "_king_pos", "_priest_count")

    def __init__(self, board: Board, current: Color) -> None:
        self._board = board
        self._current = current
        self._legal: Dict[Color, List[Move]] = {Color.WHITE: [], Color.BLACK: []}
        self._king_pos: Dict[Color, Tuple[int, int, int]] = {}
        self._priest_count: Dict[Color, int] = {Color.WHITE: 0, Color.BLACK: 0}
        self._full_rebuild()

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------
    def legal_moves(self, color: Color) -> List[Move]:
        """Return cached legal moves for side `color`."""
        return self._legal[color]

    def apply_move(self, mv: Move, color: Color) -> None:
        """Make move on internal mirror board and update cache."""
        self._board.apply_move(mv)          # your existing tensor helper
        self._current = color.opposite()
        self._refresh_counts()
        self._rebuild_if_needed(color)      # full rebuild only on priest / king change
        self._current = color.opposite()

    def undo_move(self, mv: Move, color: Color) -> None:
        """Undo move (assumes perfect symmetry of apply/undo)."""
        self._board.undo_move(mv)           # you’ll add this tiny helper
        self._current = color
        self._refresh_counts()
        self._rebuild_if_needed(color)
        self._current = color

    # ----------------------------------------------------------
    # internals
    # ----------------------------------------------------------
    def _full_rebuild(self) -> None:
        """Brute-force rebuild both sides."""
        for col in (Color.WHITE, Color.BLACK):
            self._legal[col] = self._generate_side(col)

    def _generate_side(self, color: Color) -> List[Move]:
        """Generate true legal moves for one side."""
        legal: List[Move] = []
        # temporarily set state attributes that generators expect
        tmp_current = self._current
        self._current = color
        for mv in generate_pseudo_legal_moves(self._board, color):
            self._board.apply_move(mv)
            if not king_in_check(self._board, color, color):
                legal.append(mv)
            self._board.undo_move(mv)
        self._current = tmp_current
        return legal

    def _refresh_counts(self) -> None:
        """Re-scan king & priest counts."""
        for col in (Color.WHITE, Color.BLACK):
            self._priest_count[col] = 0
            self._king_pos[col] = None
            for c, p in self._board.list_occupied():
                if p.color == col:
                    if p.ptype == PieceType.PRIEST:
                        self._priest_count[col] += 1
                    elif p.ptype == PieceType.KING:
                        self._king_pos[col] = c

    def _rebuild_if_needed(self, side_that_moved: Color) -> None:
        """Full rebuild only if priest or king count changed for either side."""
        old_p = self._priest_count.copy()
        old_k = self._king_pos.copy()
        self._refresh_counts()
        if (old_p != self._priest_count) or (old_k != self._king_pos):
            self._full_rebuild()


# ------------------------------------------------------------------
# global singleton helpers
# ------------------------------------------------------------------
def init_cache(board: Board, current: Color) -> None:
    global _move_cache
    _move_cache = MoveCache(board, current)


def get_cache() -> MoveCache:
    if _move_cache is None:
        raise RuntimeError("MoveCache not initialized")
    return _move_cache
