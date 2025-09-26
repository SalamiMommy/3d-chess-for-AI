#game3d/cache/effectscache/blackholesuckccahe.py
"""Incremental cache for Black-Hole suck pull map."""

from __future__ import annotations
from typing import Dict, Tuple, Optional
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.blackholesuck import suck_candidates
from game3d.movement.movepiece import Move

class BlackHoleSuckCache:
    __slots__ = ("_pull_map",)

    def __init__(self) -> None:
        self._pull_map: Dict[Color, Dict[Tuple[int, int, int], Tuple[int, int, int]]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }

    # ---------- public ----------
    def pull_map(self, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        return self._pull_map[controller]

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    # ---------- internals ----------
    def _rebuild(self, board: Board) -> None:
        for col in (Color.WHITE, Color.BLACK):
            self._pull_map[col] = suck_candidates(board, col)
