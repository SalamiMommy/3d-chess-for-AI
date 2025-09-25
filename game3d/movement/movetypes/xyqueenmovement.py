# game3d/movement/movetypes/xyqueenmovement.py

"""3D XY-Queen movement logic — moves like 2D queen + 1-step king in XY-plane (Z fixed)."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import (
    slide_along_directions,
    validate_piece_at
)

def generate_xy_queen_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    start = (x, y, z)
    if not validate_piece_at(state, start, PieceType.XYQUEEN):
        return []
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (1, 1, 0), (-1, -1, 0),
        (1, -1, 0), (-1, 1, 0)
    ]
    raw_moves = slide_along_directions(
        state=state,
        start=start,
        directions=directions,
        allow_capture=True,
        allow_self_block=False
    )
    moves = [move for move in raw_moves if move.to_coord[2] == z]
    return moves
