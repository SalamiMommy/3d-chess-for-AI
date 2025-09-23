"""Incremental cache for White-Hole push map."""

from __future__ import annotations
from typing import Dict, Tuple, Optional
from pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.whiteholepush import push_candidates
from game.move import Move


class WhiteHolePushCache:
    __slots__ = ("_push_map", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        self._push_map: Dict[Color, Dict[Tuple[int, int, int], Tuple[int, int, int]]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }
        self._rebuild()

    # ---------- public ----------
    def push_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """Return {enemy_square: push_target} for controller's turn-end."""
        return self._push_map[controller]

    def apply_move(self, mv: Move, mover: Color) -> None:
        self._board.apply_move(mv)
        self._rebuild()

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._rebuild()

    # ---------- internals ----------
    def _rebuild(self) -> None:
        for col in (Color.WHITE, Color.BLACK):
            self._push_map[col] = push_candidates(self._board, col)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_push_cache: Optional[WhiteHolePushCache] = None


def init_white_hole_push_cache(board: Board) -> None:
    global _push_cache
    _push_cache = WhiteHolePushCache(board)


def get_white_hole_push_cache() -> WhiteHolePushCache:
    if _push_cache is None:
        raise RuntimeError("WhiteHolePushCache not initialised")
    return _push_cache
