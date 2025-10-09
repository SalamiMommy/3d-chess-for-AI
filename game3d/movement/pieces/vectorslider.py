"""
Vector-Slider — 152 primitive directions ≤ 3 via slider movement (consolidated).
Exports:
  generate_vector_slider_moves(cache, color, x, y, z) -> list[Move]
  (decorated) vectorslider_dispatcher(state, x, y, z) -> list[Move]
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING
from math import gcd
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

# 152 primitive directions |dx|,|dy|,|dz| ≤ 3
def _vector_dirs() -> np.ndarray:
    dirs = set()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if dx == dy == dz == 0:
                    continue
                g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
                dirs.add((dx // g, dy // g, dz // g))
    return np.array(list(dirs), dtype=np.int8)

VECTOR_DIRECTIONS = _vector_dirs()

def generate_vector_slider_moves(cache: CacheManager,
                                 color: Color,
                                 x: int, y: int, z: int) -> List[Move]:
    """Single call to slider engine."""
    return get_slider_generator().generate_moves(
        piece_type='vector_slider',
        pos=(x, y, z),
        color=color.value,
        max_distance=8,
        cache_manager=cache,          # ← REQUIRED
        directions=VECTOR_DIRECTIONS
    )

@register(PieceType.VECTORSLIDER)
def vectorslider_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_vector_slider_moves(state.cache, state.color, x, y, z)

__all__ = ['generate_vector_slider_moves']
