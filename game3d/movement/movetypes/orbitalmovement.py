"""Orbital piece — Manhattan-distance-4 jumps via the jump engine."""

from __future__ import annotations

from typing import List
import numpy as np

from game3d.pieces.enums import Color
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.cache.manager import CacheManager

# 66 offsets where |dx|+|dy|+|dz| == 4
_ORBITAL_DIRECTIONS: np.ndarray = np.array([
    (dx, dy, dz)
    for dx in range(-4, 5)
    for dy in range(-4, 5)
    for dz in range(-4, 5)
    if abs(dx) + abs(dy) + abs(dz) == 4
], dtype=np.int8)               # shape (66, 3)

def generate_orbital_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate all legal Orbiter moves from (x, y, z) using the jump engine."""
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        position=(x, y, z),
        directions=_ORBITAL_DIRECTIONS,
        allow_capture=True,
        use_amd=True          # keep GPU path when available
    )

# Optional helper for external consumers
def get_orbital_directions() -> np.ndarray:
    return _ORBITAL_DIRECTIONS.copy()
