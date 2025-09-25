"""Incremental cache for White-Hole push map."""

from __future__ import annotations
from typing import Dict, Tuple, Optional
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.whiteholepush import push_candidates
from game3d.movement.movepiece import Move

class WhiteHolePushCache:
    __slots__ = ("_push_map",)

    def __init__(self) -> None:
        self._push_map: Dict[Color, Dict[Tuple[int, int, int], Tuple[int, int, int]]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }

    # ---------- public ----------
    def push_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self._push_map[controller]

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    # ---------- internals ----------
    def _rebuild(self, board: Board) -> None:
        for col in (Color.WHITE, Color.BLACK):
            self._push_map[col] = push_candidates(board, col)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_push_cache: Optional[WhiteHolePushCache] = None

def init_white_hole_push_cache() -> None:
    global _push_cache
    _push_cache = WhiteHolePushCache()

def get_white_hole_push_cache() -> WhiteHolePushCache:
    if _push_cache is None:
        raise RuntimeError("WhiteHolePushCache not initialised")
    return _push_cache
