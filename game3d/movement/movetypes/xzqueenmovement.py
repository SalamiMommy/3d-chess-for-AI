"""3D XZ-Queen movement logic — 2-D queen in XZ-plane via slidermovement."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.cache.manager import OptimizedCacheManager

# 8 directions in XZ-plane (Y fixed)
XZ_QUEEN_DIRECTIONS = np.array([
    (1, 0, 0), (-1, 0, 0),   # X axis
    (0, 0, 1), (0, 0, -1),   # Z axis
    (1, 0, 1), (1, 0, -1),   # XZ diagonals
    (-1, 0, 1), (-1, 0, -1)
], dtype=np.int8)

def generate_xz_queen_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal XZ-QUEEN moves from (x, y, z)."""
    engine = get_slider_generator(cache)
    return engine.generate(
        color=color,
        ptype=PieceType.XZQUEEN,   # <-- NEW
        pos=(x, y, z),
        directions=XZ_QUEEN_DIRECTIONS,
        max_steps=8,
       
        
        
        
    )
