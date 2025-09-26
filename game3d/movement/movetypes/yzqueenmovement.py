# game3d/movement/movetypes/yzqueenmovement.py

"""3D YZ-Queen movement logic — moves like 2D queen + 1-step king in YZ-plane (X fixed)."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import (
    slide_along_directions,
    validate_piece_at
)


def generate_yz_queen_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Generate all legal YZ-QUEEN moves from (x, y, z).
    Moves like a 2D queen in the YZ-plane — X remains fixed.
    """
    start = (x, y, z)

    if not validate_piece_at(state, start, expected_type=PieceType.YZQUEEN):
        return []

    # Define 8 YZ-plane directions (dx = 0 always)
    directions = [
        (0, 1, 0), (0, -1, 0),   # along Y
        (0, 0, 1), (0, 0, -1),   # along Z
        (0, 1, 1), (0, -1, -1),  # diagonal YZ
        (0, 1, -1), (0, -1, 1)
    ]

    raw_moves = slide_along_directions(
        state=state,
        start=start,
        directions=directions,
        allow_capture=True,
        allow_self_block=False
    )

    # Enforce X remains fixed
    moves = [
        move for move in raw_moves
        if move.to_coord[0] == x
    ]

    return moves
