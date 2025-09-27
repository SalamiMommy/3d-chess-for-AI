# game3d/movement/movetypes/yzqueenmovement.py
"""3D YZ-Queen movement logic — moves like 2D queen in YZ-plane (X fixed)."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at

def generate_yz_queen_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal YZ-QUEEN moves from (x, y, z)."""
    start = (x, y, z)

    if not validate_piece_at(state, start, pos, expected_type=PieceType.YZQUEEN):
        return []

    # 8 directions in YZ-plane (X fixed)
    directions = [
        (0, 1, 0), (0, -1, 0),   # Y axis
        (0, 0, 1), (0, 0, -1),   # Z axis
        (0, 1, 1), (0, 1, -1),   # YZ diagonals
        (0, -1, 1), (0, -1, -1)
    ]

    return slide_along_directions(
        state=state,
        start=start,
        directions=directions,
        allow_capture=True,
        allow_self_block=False
    )
