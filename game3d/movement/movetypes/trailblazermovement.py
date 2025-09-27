#game3d/movement/movetypes/trailblazermovement.py
"""Trailblazer — rook moves capped at 3 squares per ray; marks full path for aura."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import slide_along_directions, validate_piece_at

ROOK_DIRECTIONS_3D = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

def generate_trailblazer_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all 3-step rook slides from (x, y, z)."""
    start = (x, y, z)

    if not validate_piece_at(state, start, pos, expected_type=PieceType.TRAILBLAZER):
        return []

    # Pure move generation — NO SIDE EFFECTS
    return slide_along_directions(
        state=state,
        start=start,
        directions=ROOK_DIRECTIONS_3D,
        allow_capture=True,
        allow_self_block=False,
        max_steps=3,
        edge_only=False,
    )
