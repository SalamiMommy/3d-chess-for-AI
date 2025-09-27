# game3d/movement/movetypes/trigonalbishopmovement.py
"""3D Trigonal-Bishop move generation — pure movement rules, no registration."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at

# 8 true 3D space diagonals (equal step on all three axes)
TRIGONAL_BISHOP_DIRECTIONS = [
    ( 1,  1,  1),
    ( 1,  1, -1),
    ( 1, -1,  1),
    ( 1, -1, -1),
    (-1,  1,  1),
    (-1,  1, -1),
    (-1, -1,  1),
    (-1, -1, -1),
]

def generate_trigonal_bishop_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all legal trigonal-bishop moves from (x, y, z)."""
    pos = (x, y, z)
    if not validate_piece_at(state.board, state.color, pos, PieceType.TRIGONALBISHOP):
        return []

    return slide_along_directions(
        state,
        start=pos,
        directions=TRIGONAL_BISHOP_DIRECTIONS,
        allow_capture=True,
        allow_self_block=False
    )
