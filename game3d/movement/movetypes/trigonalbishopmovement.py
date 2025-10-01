"""3D Trigonal-Bishop move generation — space-diagonal rays via slidermovement."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_integrated_movement_generator
from game3d.cache.manager import OptimizedCacheManager

# 8 true 3-D space diagonals (equal step on all three axes)
TRIGONAL_BISHOP_DIRECTIONS = np.array([
    ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
    (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1),
], dtype=np.int8)

def generate_trigonal_bishop_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int,
    y: int,
    z: int
) -> List[Move]:
    engine = get_integrated_movement_generator(cache)
    return engine.generate_sliding_moves(
        color=color,
        piece_type=PieceType.TRIGONALBISHOP,   # <-- NEW
        position=(x, y, z),
        directions=TRIGONAL_BISHOP_DIRECTIONS,
        max_steps=9,
        allow_capture=True,
        allow_self_block=False,
        use_symmetry=True,
        use_amd=True
    )
