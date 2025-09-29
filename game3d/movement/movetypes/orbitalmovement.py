"""Orbital piece — jumps to any square with Manhattan distance 4.
Pure movement logic — no registration, no dispatcher.
"""

from __future__ import annotations
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import jump_to_targets, validate_piece_at
from game3d.cache.manager import OptimizedCacheManager

# Precomputed offsets: all (dx,dy,dz) where |dx|+|dy|+|dz| == 4
_ORBITAL_OFFSETS = [
    (dx, dy, dz)
    for dx in range(-4, 5)
    for dy in range(-4, 5)
    for dz in range(-4, 5)
    if abs(dx) + abs(dy) + abs(dz) == 4
]
# Total: 66 offsets (3D octahedron surface)

def generate_orbital_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal Orbiter moves from (x, y, z)."""
    pos = (x, y, z)
    if not validate_piece_at(cache, color, pos, PieceType.ORBITER):
        return []

    return jump_to_targets(
        cache=cache,  # ✅ FIXED: cache, not board
        color=color,
        start=pos,
        offsets=_ORBITAL_OFFSETS,
        allow_capture=True,
        allow_self_block=False
    )
