"""3D Rook move generation — orthogonal rays via slidermovement engine."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.cache.manager import OptimizedCacheManager

# 6 orthogonal directions (±X, ±Y, ±Z)
ROOK_DIRECTIONS_3D = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
], dtype=np.int8)

def generate_rook_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int,
    y: int,
    z: int,
    max_steps: int = 9
) -> List[Move]:
    """Generate all legal rook moves from (x, y, z) up to max_steps."""
    engine = get_slider_generator(cache)
    return engine.generate(
        color=color,
        ptype=PieceType.ROOK,   # <-- NEW
        pos=(x, y, z),
        directions=ROOK_DIRECTIONS_3D,
        max_steps=max_steps,
       
        
        
        
    )
