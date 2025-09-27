# game3d/movement/movetypes/queenmovement.py
"""3D Queen move generation logic — pure movement rules, no registration."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at

# Queen directions: 26 total (6 orthogonal + 12 face diagonals + 8 space diagonals)
ORTHOGONAL_DIRECTIONS = [
    (1, 0, 0), (-1, 0, 0),  # X
    (0, 1, 0), (0, -1, 0),  # Y
    (0, 0, 1), (0, 0, -1)   # Z
]

FACE_DIAGONALS = [
    # XY plane
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    # XZ plane
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    # YZ plane
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
]

SPACE_DIAGONALS = [
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
]

QUEEN_DIRECTIONS_3D = ORTHOGONAL_DIRECTIONS + FACE_DIAGONALS + SPACE_DIAGONALS

def generate_queen_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal queen moves from (x, y, z)."""
    pos = (x, y, z)
    if not validate_piece_at(state.board, state.color, pos, PieceType.QUEEN):
        return []

    return slide_along_directions(
        state,
        start=pos,
        directions=QUEEN_DIRECTIONS_3D,
        allow_capture=True,
        allow_self_block=False
    )
