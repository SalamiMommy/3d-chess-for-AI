# game3d/movement/movetypes/trigonalbishopmovement.py
"""3D Trigonal-Bishop move generation — pure movement rules, no registration."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at
from game3d.movement.movepiece import Move  # Ensure Move is available

# 8 true 3D space diagonals (equal step on all three axes)
TRIGONAL_BISHOP_DIRECTIONS = np.array([
    ( 1,  1,  1),
    ( 1,  1, -1),
    ( 1, -1,  1),
    ( 1, -1, -1),
    (-1,  1,  1),
    (-1,  1, -1),
    (-1, -1,  1),
    (-1, -1, -1),
])

def generate_trigonal_bishop_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal trigonal-bishop moves from (x, y, z)."""
    pos = (x, y, z)

    # Validate piece at starting position
    if not validate_piece_at(board, color, pos, PieceType.TRIGONALBISHOP):
        return []

    return slide_along_directions(
        board=board,
        color=color,
        start=pos,
        directions=TRIGONAL_BISHOP_DIRECTIONS,
        allow_capture=True,
        allow_self_block=False
    )
