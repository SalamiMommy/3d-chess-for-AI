"""
Vector-Slider — 152 primitive directions ≤ 3 via slider movement (consolidated).
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
from math import gcd
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_moves
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

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

def generate_vector_slider_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    return generate_moves(
        piece_type='vector_slider',
        pos=(x, y, z),
        color=color,
        max_distance=8,
        directions=VECTOR_DIRECTIONS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

@register(PieceType.VECTORSLIDER)
def vectorslider_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_vector_slider_moves(state.cache_manager, state.color, x, y, z)  # FIXED: Use cache_manager

__all__ = ['generate_vector_slider_moves']
