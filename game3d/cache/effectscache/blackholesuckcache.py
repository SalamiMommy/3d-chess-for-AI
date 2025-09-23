"""Incremental cache for Black-Hole suck pull map."""

from __future__ import annotations
from typing import Dict, Tuple, Optional
from pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.blackholesuck import suck_candidates
from game.move import Move

class BlackHoleSuckCache:
    __slots__ = ("_pull_map", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        self._pull_map: Dict[Color, Dict[Tuple[int, int, int], Tuple[int, int, int]]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }
        self._rebuild()

    def pull_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self._pull_map[controller]

    def apply_move(self, mv: Move, mover: Color) -> None:
        self._board.apply_move(mv)
        self._rebuild()

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._rebuild()

    def _rebuild(self) -> None:
        for col in (Color.WHITE, Color.BLACK):
            self._pull_map[col] = suck_candidates(self._board, col)

_suck_cache: Optional[BlackHoleSuckCache] = None

def init_black_hole_suck_cache(board: Board) -> None:
    global _suck_cache
    _suck_cache = BlackHoleSuckCache(board)

def get_black_hole_suck_cache() -> BlackHoleSuckCache:
    if _suck_cache is None:
        raise RuntimeError("BlackHoleSuckCache not initialised")
    return _suck_cache
