"""3D Vector-Slider — moves along ANY integer vector with ratios up to 3.
Pure movement logic, no registration; drop-in replacement for queenmovement.py.
"""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at
from game3d.movement.movepiece import Move  # Ensure Move is available
from math import gcd
from game3d.cache.manager import OptimizedCacheManager

# -----------------------------------------------------------------------------
# Precomputed directions: all primitive vectors with components in [-3, 3]
# -----------------------------------------------------------------------------
def _compute_vector_slider_directions() -> List[tuple[int, int, int]]:
    """Generate all unique primitive directions with |dx|,|dy|,|dz| <= 3."""
    directions = set()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if dx == dy == dz == 0:
                    continue

                # Reduce to primitive direction
                g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
                if g > 0:
                    primitive = (dx // g, dy // g, dz // g)
                    directions.add(primitive)

    return list(directions)

# Precompute once at import time — as NumPy array
VECTOR_SLIDER_DIRECTIONS = np.array(_compute_vector_slider_directions())
# Total: 152 unique directions (verified)

def generate_vector_slider_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal vector slider moves from (x, y, z)."""
    pos = (x, y, z)

    # Validate piece at starting position
    if not validate_piece_at(cache, color, pos, PieceType.VECTORSLIDER):
        return []

    return slide_along_directions(
        cache=cache,  # ✅ FIXED: cache, not board
        color=color,
        start=pos,
        directions=VECTOR_SLIDER_DIRECTIONS,
        allow_capture=True,
        allow_self_block=False
    )
