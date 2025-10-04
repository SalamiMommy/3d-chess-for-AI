"""Echo piece — radius-2 sphere projected 3 steps away in every ±x/±y/±z direction.
Zero-redundancy implementation via the existing jump engine.
"""

from __future__ import annotations

from typing import List
import numpy as np

from game3d.pieces.enums import Color
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.cache.manager import CacheManager

# 8 anchors: (±3, ±3, ±3)
_ANCHOR_OFFSETS = np.array([
    (dx, dy, dz)
    for dx in (-3, 3)
    for dy in (-3, 3)
    for dz in (-3, 3)
], dtype=np.int8)                    # shape (8, 3)

# 32 bubble offsets: radius ≤ 2, origin excluded
_BUBBLE_OFFSETS = np.array([
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if dx*dx + dy*dy + dz*dz <= 4 and not (dx == dy == dz == 0)
], dtype=np.int8)                    # shape (32, 3)

# 256 final directions: anchor + bubble
_ECHO_DIRECTIONS: np.ndarray = (
    _ANCHOR_OFFSETS[:, None, :] + _BUBBLE_OFFSETS[None, :, :]
).reshape(-1, 3)                     # shape (256, 3)

def generate_echo_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate all legal Echo moves from (x, y, z) using the jump engine."""
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=_ECHO_DIRECTIONS,
       
                  # keep GPU path when available
    )

# Optional helpers for external consumers
def get_echo_directions() -> np.ndarray:
    return _ECHO_DIRECTIONS.copy()
