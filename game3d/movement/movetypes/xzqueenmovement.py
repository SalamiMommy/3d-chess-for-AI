"""3D XZ-Queen movement logic — moves like 2D queen in XZ-plane (Y fixed)."""

import numpy as np  # needed for slide_along_directions
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at
from game3d.movement.movepiece import Move

def generate_xz_queen_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal XZ-QUEEN moves from (x, y, z)."""
    start = (x, y, z)

    # Validate piece at start
    if not validate_piece_at(board, color, start, PieceType.XZQUEEN):
        return []

    # 8 directions in XZ-plane (Y fixed)
    directions = np.array([
        (1, 0, 0), (-1, 0, 0),   # X axis
        (0, 0, 1), (0, 0, -1),   # Z axis
        (1, 0, 1), (1, 0, -1),   # XZ diagonals
        (-1, 0, 1), (-1, 0, -1)
    ])

    return slide_along_directions(
        board=board,
        color=color,
        start=start,
        directions=directions,
        allow_capture=True,
        allow_self_block=False
    )
