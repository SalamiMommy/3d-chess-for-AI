"""3D King move generation logic — pure movement rules, no registration."""

import numpy as np
from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at
from game3d.movement.movepiece import Move

# King moves 1 step in any direction → 26 neighbors
KING_DIRECTIONS_3D = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)  # exclude (0,0,0)
])

def generate_king_moves(board, color: Color, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal king moves from (x, y, z).
    King = single-step queen → 26 directions, max 1 step.

    Does NOT include castling (implement separately if desired).
    """
    pos = (x, y, z)

    # Validate piece exists and is correct type/color
    if not validate_piece_at(board, color, pos, PieceType.KING):
        return []

    # Reuse slide logic — but limit to 1 step
    return slide_along_directions(
        board=board,
        color=color,
        start=pos,
        directions=KING_DIRECTIONS_3D,
        allow_capture=True,      # Kings can capture
        allow_self_block=False,  # Kings cannot land on friendly pieces
        max_steps=1              # 👑 King only moves 1 square!
    )
