"""3D Vector-Slider — moves along ANY integer vector with ratios up to 3 via slidermovement."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator
from math import gcd
from game3d.cache.manager import OptimizedCacheManager

# ---------------------------------------------------------------------------
# Pre-compute primitive directions with |dx|,|dy|,|dz| <= 3
# ---------------------------------------------------------------------------
# Pre-compute primitive directions with |dx|,|dy|,|dz| <= 3
def _compute_vector_slider_directions() -> np.ndarray:
    directions = set()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if dx == dy == dz == 0:
                    continue
                g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
                if g:
                    directions.add((dx // g, dy // g, dz // g))
    return np.array(list(directions), dtype=np.int8)

# Compute once at module load time
VECTOR_SLIDER_DIRECTIONS = _compute_vector_slider_directions()  # 152 dirs

# ---------------------------------------------------------------------------
# Public API — drop-in replacement
# ---------------------------------------------------------------------------
def generate_vector_slider_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    # Validate inputs
    if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
        raise ValueError(f"Invalid position: ({x}, {y}, {z})")

    if color not in (Color.WHITE, Color.BLACK):
        raise ValueError(f"Invalid color: {color}")

    # Check if there's a piece at the given position
    piece = cache.piece_cache.get((x, y, z))
    if piece is None:
        return []

    if piece.color != color:
        return []

    if piece.ptype != PieceType.VECTORSLIDER:
        return []

    engine = get_slider_generator()  # FIXED: Removed argument
    return engine.generate_moves(   # FIXED: Changed method name
        piece_type='vector_slider',  # Added piece_type
        pos=(x, y, z),
        board_occupancy=cache.occupancy.mask
,  # Added board_occupancy
        color=color.value if isinstance(color, Color) else color,  # Convert to int
        max_distance=8,  # Changed from max_steps
    )
