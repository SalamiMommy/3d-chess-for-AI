"""Trailblazing – marks slid squares for last 3 moves; enemy step → +1 counter; 3 = removal (king spared if priests)."""

from __future__ import annotations
from typing import List, Tuple, Dict, Set
from collections import deque
from game3d.pieces.enums import Color, PieceType
from game3d.common.protocols import BoardProto


class TrailblazeRecorder:
    """Per-trailblazer FIFO of last 3 slid-square sets."""
    __slots__ = ("_history",)

    def __init__(self) -> None:
        self._history: deque[Set[Tuple[int, int, int]]] = deque(maxlen=3)

    # ---------- public ----------
    def add_trail(self, squares: Set[Tuple[int, int, int]]) -> None:
        self._history.append(squares)

    def current_trail(self) -> Set[Tuple[int, int, int]]:
        """Union of last 3 trails."""
        out: Set[Tuple[int, int, int]] = set()
        for s in self._history:
            out.update(s)
        return out


def squares_to_mark(board: BoardProto, controller: Color) -> Dict[Tuple[int, int, int], TrailblazeRecorder]:
    """Return {trailblazer_coord: its recorder} for all friendly Trailblazers."""
    out: Dict[Tuple[int, int, int], TrailblazeRecorder] = {}
    for coord, piece in board.list_occupied():
        if piece.color == controller and piece.ptype == PieceType.TRAILBLAZER:
            out[coord] = TrailblazeRecorder()
    return out
