"""Nebula piece — sphere-radius-3 jumps via the zero-redundancy jump engine."""

from __future__ import annotations

from typing import List
import numpy as np

from game3d.pieces.enums import Color
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.cache.manager import CacheManager

# 122 pre-computed offsets (dx,dy,dz) with dx²+dy²+dz² ≤ 9, excluding (0,0,0)
_NEBULA_OFFSETS: np.ndarray = np.array([
    (dx, dy, dz)
    for dx in range(-3, 4)
    for dy in range(-3, 4)
    for dz in range(-3, 4)
    if dx*dx + dy*dy + dz*dz <= 9 and not (dx == dy == dz == 0)
], dtype=np.int8)          # shape (122, 3)

def generate_nebula_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate all legal Nebula moves from (x, y, z) using the jump engine."""
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=_NEBULA_OFFSETS,
       
                  # keep GPU path when available
    )

# Optional helper for external consumers
def get_nebula_offsets() -> np.ndarray:
    return _NEBULA_OFFSETS.copy()
