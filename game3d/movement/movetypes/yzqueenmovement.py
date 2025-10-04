"""3D YZ-Queen movement logic — 2-D queen in YZ-plane via slidermovement."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.cache.manager import OptimizedCacheManager

# 8 directions in YZ-plane (X fixed)
YZ_QUEEN_DIRECTIONS = np.array([
    (0, 1, 0), (0, -1, 0),   # Y axis
    (0, 0, 1), (0, 0, -1),   # Z axis
    (0, 1, 1), (0, 1, -1),   # YZ diagonals
    (0, -1, 1), (0, -1, -1)
], dtype=np.int8)

def generate_yz_queen_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal YZ-QUEEN moves from (x, y, z)."""
    engine = get_slider_generator(cache)
    return engine.generate(
        color=color,
        ptype=PieceType.YZQUEEN,   # <-- NEW
        pos=(x, y, z),
        directions=YZ_QUEEN_DIRECTIONS,
        max_steps=8,
       
        
        
        
    )
