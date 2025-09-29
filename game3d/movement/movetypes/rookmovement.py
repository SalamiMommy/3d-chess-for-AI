"""3D Rook move generation logic — pure movement rules, no registration."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at
from game3d.cache.manager import OptimizedCacheManager

# 6 orthogonal directions (±X, ±Y, ±Z)
ROOK_DIRECTIONS_3D = np.array([
    (1, 0, 0), (-1, 0, 0),  # X axis
    (0, 1, 0), (0, -1, 0),  # Y axis
    (0, 0, 1), (0, 0, -1)   # Z axis
])

def generate_rook_moves(
    cache: OptimizedCacheManager,  # ← CHANGED: board → cache
    color: Color,
    x: int,
    y: int,
    z: int
) -> List['Move']:
    """Generate all legal rook moves from (x, y, z)."""
    pos = (x, y, z)

    if not validate_piece_at(cache, color, pos, PieceType.ROOK):  # ← cache, not board
        return []

    return slide_along_directions(
        cache=cache,  # ← FIXED: cache, not board
        color=color,
        start=pos,
        directions=ROOK_DIRECTIONS_3D,
        allow_capture=True,
        allow_self_block=False
    )
