# game3d/movement/movetypes/xzqueenmovement.py

"""3D XZ-Queen movement logic — moves like 2D queen + 1-step king in XZ-plane (Y fixed)."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.pathvalidation import (
    slide_along_directions,
    validate_piece_at
)


def generate_xz_queen_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal XZ-QUEEN moves from (x, y, z).
    Moves like a 2D queen in the XZ-plane — Y remains fixed.
    """
    start = (x, y, z)

    if not validate_piece_at(state, start, expected_type=PieceType.XZ_QUEEN):
        return []

    # Define 8 XZ-plane directions (dy = 0 always)
    directions = [
        (1, 0, 0), (-1, 0, 0),   # along X
        (0, 0, 1), (0, 0, -1),   # along Z
        (1, 0, 1), (-1, 0, -1),  # diagonal XZ
        (1, 0, -1), (-1, 0, 1)
    ]

    raw_moves = slide_along_directions(
        state=state,
        start=start,
        directions=directions,
        allow_capture=True,
        allow_self_block=False
    )

    # Enforce Y remains fixed
    moves = [
        move for move in raw_moves
        if move.to_coord[1] == y
    ]

    return moves
