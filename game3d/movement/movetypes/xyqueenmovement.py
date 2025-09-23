# game3d/movement/movetypes/xyqueenmovement.py

"""3D XY-Queen movement logic — moves like 2D queen + 1-step king in XY-plane (Z fixed)."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.pathvalidation import (
    slide_along_directions,
    validate_piece_at
)
from common import in_bounds


def generate_xy_queen_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal XY-QUEEN moves from (x, y, z).
    Moves like a 2D queen in the XY-plane — Z remains fixed.
    """
    start = (x, y, z)

    if not validate_piece_at(state, start, expected_type=PieceType.XY_QUEEN):
        return []

    # Define 8 XY-plane directions (dz = 0 always)
    directions = [
        (1, 0, 0), (-1, 0, 0),   # horizontal
        (0, 1, 0), (0, -1, 0),   # vertical
        (1, 1, 0), (-1, -1, 0),  # diagonal
        (1, -1, 0), (-1, 1, 0)
    ]

    # Use centralized slider
    raw_moves = slide_along_directions(
        state=state,
        start=start,
        directions=directions,
        allow_capture=True,
        allow_self_block=False
    )

    # Enforce Z remains fixed
    moves = [
        move for move in raw_moves
        if move.to_coord[2] == z
    ]

    return moves
