"""Knight (3,1) leaper — 3-axis 3-D knight."""

from typing import List, Tuple
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import validate_piece_at
from game3d.common.common import in_bounds, add_coords


VECTORS_31 = [
    (dx * 3, dy * 1, dz * 1)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
] + [
    (dx * 1, dy * 3, dz * 1)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
] + [
    (dx * 1, dy * 1, dz * 3)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
]

VECTORS_31 = list(set(VECTORS_31))


def generate_knight31_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Generate all (3,1) knight leaps."""
    start = (x, y, z)
    if not validate_piece_at(state, start, expected_type=PieceType.KNIGHT31):
        return []

    moves: List[Move] = []
    board = state.board
    current_color = state.color

    for dx, dy, dz in VECTORS_31:
        target = add_coords(start, (dx, dy, dz))
        if not in_bounds(target):
            continue
        occupant = board.piece_at(target)
        if occupant is not None and occupant.color == current_color:
            continue
        is_capture = occupant is not None and occupant.color != current_color
        moves.append(Move(start, target, is_capture=is_capture))

    return moves
