"""3D XY-Queen movement logic — 2-D queen in XY-plane via slidermovement."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_integrated_movement_generator
from game3d.cache.manager import OptimizedCacheManager

# 8 directions in XY-plane (Z fixed)
XY_QUEEN_DIRECTIONS = np.array([
    (1, 0, 0), (-1, 0, 0),   # X axis
    (0, 1, 0), (0, -1, 0),   # Y axis
    (1, 1, 0), (1, -1, 0),   # XY diagonals
    (-1, 1, 0), (-1, -1, 0)
], dtype=np.int8)

def generate_xy_queen_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal XY-QUEEN moves from (x, y, z)."""
    engine = get_integrated_movement_generator(cache)
    return engine.generate_sliding_moves(
        color=color,
        piece_type=PieceType.XYQUEEN,   # <-- NEW
        position=(x, y, z),
        directions=XY_QUEEN_DIRECTIONS,
        max_steps=9,
        allow_capture=True,
        allow_self_block=False,
        use_symmetry=True,
        use_amd=True
    )
