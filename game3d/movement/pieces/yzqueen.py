"""
YZ-Queen: 8 slider rays in YZ-plane + full 3-D king hop (26 directions, 1 step).
Now uses the **hot slider kernel** for both parts.
Exports:
  generate_yz_queen_moves(cache, color, x, y, z) -> list[Move]
  (decorated) yz_queen_dispatcher(state, x, y, z) -> list[Move]
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import get_slider_generator

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

# 8 directions confined to the YZ-plane (X fixed)
_YZ_SLIDER_DIRS = np.array([
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
    (0, 1, 1), (0, 1, -1),
    (0, -1, 1), (0, -1, -1)
], dtype=np.int8)

# 26 one-step king directions (3-D)
_KING_3D_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

def generate_yz_queen_moves(cache: CacheManager,
                            color: Color,
                            x: int, y: int, z: int) -> List:
    """Slider rays (YZ-plane, 8 dirs, 8 steps) + king hop (26 dirs, 1 step)."""
    engine = get_slider_generator()

    # 1.  Long-range YZ-plane slides
    slider_moves = engine.generate_moves(
        piece_type='yz_queen',
        pos=(x, y, z),
        color=color.value,
        max_distance=8,
        cache_manager=cache,          # ← REQUIRED keyword-only argument
        directions=_YZ_SLIDER_DIRS
    )

    # 2.  One-step king hop – same kernel, max_distance=1
    king_moves = engine.generate_moves(
        piece_type='yz_queen_kinghop',   # distinct cache key
        pos=(x, y, z),
        color=color.value,
        max_distance=1,
        cache_manager=cache,          # ← REQUIRED keyword-only argument
        directions=_KING_3D_DIRS
    )

    return slider_moves + king_moves

@register(PieceType.YZQUEEN)
def yz_queen_move_dispatcher(state, x: int, y: int, z: int) -> List:
    return generate_yz_queen_moves(state.cache, state.color, x, y, z)

__all__ = ['generate_yz_queen_moves']
