"""Knight (3,2) leaper — 3-axis 3-D knight."""

from typing import List, Tuple
from pieces.enums import PieceType
from game.state import GameState
from game.move import Move
from game3d.movement.pathvalidation import validate_piece_at
from common import in_bounds, add_coords


VECTORS_32 = [
    (dx * 3, dy * 2, dz * 1)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
] + [
    (dx * 3, dy * 1, dz * 2)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
] + [
    (dx * 2, dy * 3, dz * 1)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
] + [
    (dx * 1, dy * 3, dz * 2)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
] + [
    (dx * 2, dy * 1, dz * 3)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
] + [
    (dx * 1, dy * 2, dz * 3)
    for dx in (-1, 1)
    for dy in (-1, 1)
    for dz in (-1, 1)
]

VECTORS_32 = list(set(VECTORS_32))  # dedupe


def generate_knight32_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Generate all (3,2) knight leaps."""
    start = (x, y, z)
    if not validate_piece_at(state, start, expected_type=PieceType.KNIGHT32):
        return []

    moves: List[Move] = []
    board = state.board
    current_color = state.current

    for dx, dy, dz in VECTORS_32:
        target = add_coords(start, (dx, dy, dz))
        if not in_bounds(target):
            continue
        occupant = board.piece_at(target)
        if occupant is not None and occupant.color == current_color:
            continue  # can't land on friendly
        is_capture = occupant is not None and occupant.color != current_color
        moves.append(Move(start, target, is_capture=is_capture))

    return moves
