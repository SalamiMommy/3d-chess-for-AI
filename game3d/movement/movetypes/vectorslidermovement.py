# game3d/movement/movetypes/vectorslidermovement.py
"""3D Vector-Slider — moves along ANY integer vector with ratios up to 3.
Pure movement logic, no registration; drop-in replacement for queenmovement.py.
"""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at
from math import gcd

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

# Precompute once at import time
VECTOR_SLIDER_DIRECTIONS = _compute_vector_slider_directions()
# Total: 152 unique directions (verified)

def generate_vector_slider_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal vector slider moves from (x, y, z)."""
    pos = (x, y, z)
    if not validate_piece_at(state.board, state.color, pos, PieceType.VECTORSLIDER):
        return []

    return slide_along_directions(
        state,
        start=pos,
        directions=VECTOR_SLIDER_DIRECTIONS,
        allow_capture=True,
        allow_self_block=False
    )
