# game3d/movement/movetypes/bishopmovement.py
"""3D Bishop move generation logic — pure movement rules, no registration."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at

# Precomputed constant: 20 unique 3D diagonal directions
BISHOP_DIRECTIONS = np.array([
    # XY plane diagonals (z fixed)
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    # XZ plane diagonals (y fixed)
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    # YZ plane diagonals (x fixed)
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    # Full 3D space diagonals
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
])

def generate_bishop_moves(
    board,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate all legal bishop moves from (x, y, z)."""
    pos = (x, y, z)

    # ✅ Fixed: removed duplicate 'pos'
    if not validate_piece_at(board, color, pos, PieceType.BISHOP):
        return []

    return slide_along_directions(
        board=board,
        color=color,
        start=pos,
        directions=BISHOP_DIRECTIONS,
        allow_capture=True,
        allow_self_block=False
    )
