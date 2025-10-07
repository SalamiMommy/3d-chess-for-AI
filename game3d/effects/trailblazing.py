"""Trailblazing – marks slid squares for last 3 moves; enemy step → +1 counter; 3 = removal (king spared if priests)."""

from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional, TYPE_CHECKING
from collections import deque
from game3d.pieces.enums import Color, PieceType
from game3d.common.protocols import BoardProto

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager


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


def reconstruct_trailblazer_path(start: Tuple[int, int, int], end: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """
    Reconstruct the path a trailblazer took from start to end.
    This is a simplified version that just returns the straight-line path.
    """
    path: Set[Tuple[int, int, int]] = set()

    # Calculate direction vector
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]

    # Normalize direction to unit steps
    steps = max(abs(dx), abs(dy), abs(dz))
    if steps == 0:
        return {start}

    step_x = dx // steps if dx != 0 else 0
    step_y = dy // steps if dy != 0 else 0
    step_z = dz // steps if dz != 0 else 0

    # Build path
    current = start
    for _ in range(steps):
        path.add(current)
        current = (current[0] + step_x, current[1] + step_y, current[2] + step_z)

    # Add the end position
    path.add(end)

    return path


def extract_enemy_slid_path(start: Tuple[int, int, int], end: Tuple[int, int, int], board: BoardProto,
                          cache_manager: Optional[OptimizedCacheManager] = None) -> Set[Tuple[int, int, int]]:
    """
    Extract the path an enemy piece slid on, for trailblaze interaction.
    This is a simplified version that just returns the straight-line path.
    """
    # First, verify there's actually an enemy piece at the end position
    if cache_manager:
        piece = cache_manager.piece_cache.get(end)
    else:
        piece = board.get_piece(end)

    if piece is None:
        return set()

    # Calculate direction vector
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]

    # Normalize direction to unit steps
    steps = max(abs(dx), abs(dy), abs(dz))
    if steps == 0:
        return {start}

    step_x = dx // steps if dx != 0 else 0
    step_y = dy // steps if dy != 0 else 0
    step_z = dz // steps if dz != 0 else 0

    # Build path
    path: Set[Tuple[int, int, int]] = set()
    current = start
    for _ in range(steps):
        path.add(current)
        current = (current[0] + step_x, current[1] + step_y, current[2] + step_z)

    # Add the end position
    path.add(end)

    return path
