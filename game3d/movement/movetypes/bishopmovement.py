# game3d/movement/movetypes/bishopmovement.py
"""3D Bishop move generation — now symmetry-aware via slidermovement engine."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.cache.manager import OptimizedCacheManager

# --------------------------------------------------------------------------- #
#  Geometry owned by the piece module                                        #
# --------------------------------------------------------------------------- #
BISHOP_DIRECTIONS = np.array([
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
], dtype=np.int8)

# --------------------------------------------------------------------------- #
#  Public API — signature unchanged                                          #
# --------------------------------------------------------------------------- #
def generate_bishop_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all legal bishop moves from (x, y, z) with symmetry optimisation."""
    engine = get_slider_generator(cache)
    return engine.generate(
        color=color,
        ptype=PieceType.BISHOP,   # <-- NEW
        pos=(x, y, z),
        directions=BISHOP_DIRECTIONS,
        max_steps=8,
       
        
        
        
    )
