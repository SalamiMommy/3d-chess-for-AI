#game3d/cache/effectscache/capturefrombehindcache.py
"""Incremental cache for ‘behind’ squares per WALL."""

from __future__ import annotations
from typing import Dict, Set, Tuple
from game3d.pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.capturefrombehind import from_behind_squares
from game3d.movement.movepiece import Move

class BehindCache:
    __slots__ = ("_behind",)

    def __init__(self) -> None:
        # _behind[color][wall_sq] = set of attacker squares that are "behind" that wall
        self._behind: Dict[Color, Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]]] = {
            Color.WHITE: {},
            Color.BLACK: {},
        }

    def can_capture(self, attacker_sq: Tuple[int, int, int], wall_sq: Tuple[int, int, int], controller: Color) -> bool:
        behind_map = self._behind[controller]
        behind_set = behind_map.get(wall_sq)
        return behind_set is not None and attacker_sq in behind_set

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Only rebuild the moving player's behind map."""
        self._behind[mover] = from_behind_squares(board, mover)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        """Rebuild the moving player's behind map (since undo affects their pieces)."""
        # Note: After undo, the mover is the player who originally moved
        self._behind[mover] = from_behind_squares(board, mover)

    # Optional: Add method to rebuild both if needed (e.g., after complex effects)
    def rebuild_all(self, board: Board) -> None:
        for color in (Color.WHITE, Color.BLACK):
            self._behind[color] = from_behind_squares(board, color)

