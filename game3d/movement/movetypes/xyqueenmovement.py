"""3D XY-Queen movement logic — moves like 2D queen in XY-plane (Z fixed)."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at
from game3d.movement.movepiece import Move
from game3d.cache.manager import OptimizedCacheManager

def generate_xy_queen_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal XY-QUEEN moves from (x, y, z)."""
    start = (x, y, z)

    # Validate piece at start position
    if not validate_piece_at(cache, color, start, PieceType.XYQUEEN):
        return []

    # 8 directions in XY-plane (Z fixed)
    directions = np.array([
        (1, 0, 0), (-1, 0, 0),   # X axis
        (0, 1, 0), (0, -1, 0),   # Y axis
        (1, 1, 0), (1, -1, 0),   # XY diagonals
        (-1, 1, 0), (-1, -1, 0)
    ])

    return slide_along_directions(
        cache=cache,  # ✅ FIXED: cache, not board
        color=color,
        start=start,
        directions=directions,
        allow_capture=True,
        allow_self_block=False
    )
