"""Incremental cache for frozen enemy squares."""
#game3d/cache/effects/freezcache.py
from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.freeze import frozen_squares
from game3d.movement.movepiece import Move

class FreezeCache:
    __slots__ = ("_frozen",)

    def __init__(self) -> None:
        self._frozen: Dict[Color, Set[Tuple[int, int, int]]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

    # ---------- public ----------
    def is_frozen(self, sq: Tuple[int, int, int], victim: Color) -> bool:
        return sq in self._frozen[victim]

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._rebuild(board)

    # ---------- internals ----------
    def _rebuild(self, board: Board) -> None:
        self._frozen[Color.WHITE] = frozen_squares(board, Color.BLACK)
        self._frozen[Color.BLACK] = frozen_squares(board, Color.WHITE)


