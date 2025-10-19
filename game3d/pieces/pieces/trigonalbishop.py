"""
Trigonal-Bishop — 8 space-diagonal rays (consolidated).
Exports:
  generate_trigonal_bishop_moves(cache, color, x, y, z) -> list[Move]
  (decorated) trigonal_bishop_dispatcher(state, x, y, z) -> list[Move]
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_moves
from game3d.movement.movepiece import Move

# 8 true 3-D diagonals
_TRIGONAL_DIRS = np.array([
    ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
    (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1)
], dtype=np.int8)

def generate_trigonal_bishop_moves(cache: CacheManager,
                                   color: Color,
                                   x: int, y: int, z: int) -> List[Move]:
    """Space-diagonal slider up to board edge."""
    return generate_moves(
        piece_type='trigonalbishop',
        pos=(x, y, z),
        color=color.value,
        max_distance=8,
        directions=_TRIGONAL_DIRS,
        occupancy=cache.occupancy._occ,    # ← pass the 3-D mask instead of the cache
    )

@register(PieceType.TRIGONALBISHOP)
def trigonal_bishop_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_trigonal_bishop_moves(state.cache, state.color, x, y, z)

__all__ = ['generate_trigonal_bishop_moves']
