"""Trailblazer — rook moves capped at 3 squares per ray; marks full path for aura."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at
from game3d.movement.movepiece import Move  # Ensure Move is available
from game3d.cache.manager import OptimizedCacheManager

ROOK_DIRECTIONS_3D = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
])

def generate_trailblazer_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all 3-step rook slides from (x, y, z)."""
    start = (x, y, z)

    # Validate piece at start position
    if not validate_piece_at(cache, color, start, expected_type=PieceType.TRAILBLAZER):
        return []

    # Pure move generation — NO SIDE EFFECTS
    return slide_along_directions(
        cache=cache,  # ✅ FIXED: cache, not board
        color=color,
        start=start,
        directions=ROOK_DIRECTIONS_3D,
        allow_capture=True,
        allow_self_block=False,
        max_steps=3
    )
