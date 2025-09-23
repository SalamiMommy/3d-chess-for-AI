# game3d/movement/movetypes/xzqueenmovement.py

"""3D XZ-Queen movement logic — moves like 2D queen + 1-step king in XZ-plane (Y fixed)."""

from typing import List
from pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import (
    slide_along_directions,
    validate_piece_at
)

def generate_xzqueen_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    start = (x, y, z)
    if not validate_piece_at(state, start, PieceType.XZQUEEN):
        return []
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 0, 1), (-1, 0, -1),
        (1, 0, -1), (-1, 0, 1)
    ]
    raw_moves = slide_along_directions(
        state=state,
        start=start,
        directions=directions,
        allow_capture=True,
        allow_self_block=False
    )
    moves = [move for move in raw_moves if move.to_coord[1] == y]
    return moves
