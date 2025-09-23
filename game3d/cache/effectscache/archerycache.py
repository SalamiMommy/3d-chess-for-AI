"""Incremental cache for Archery attack targets."""

from __future__ import annotations
from typing import List, Tuple, Optional
from pieces.enums import Color
from game3d.board.board import Board
from game3d.effects.auras.archery import archery_targets
from game.move import Move


class ArcheryCache:
    __slots__ = ("_targets", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        self._targets: Dict[Color, List[Tuple[int, int, int]]] = {
            Color.WHITE: [],
            Color.BLACK: [],
        }
        self._rebuild()

    # ---------- public ----------
    def attack_targets(self, controller: Color) -> List[Tuple[int, int, int]]:
        """Current valid attack targets for controller."""
        return self._targets[controller]

    def is_valid_attack(self, sq: Tuple[int, int, int], controller: Color) -> bool:
        """Quick O(1) lookup (list is small)."""
        return sq in self._targets[controller]

    def apply_move(self, mv: Move, mover: Color) -> None:
        self._board.apply_move(mv)
        self._rebuild()   # archers may have moved

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._rebuild()

    # ---------- internals ----------
    def _rebuild(self) -> None:
        for col in (Color.WHITE, Color.BLACK):
            self._targets[col] = archery_targets(self._board, col)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_arch_cache: Optional[ArcheryCache] = None


def init_archery_cache(board: Board) -> None:
    global _arch_cache
    _arch_cache = ArcheryCache(board)


def get_archery_cache() -> ArcheryCache:
    if _arch_cache is None:
        raise RuntimeError("ArcheryCache not initialised")
    return _arch_cache
