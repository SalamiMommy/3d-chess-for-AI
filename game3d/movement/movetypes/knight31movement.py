"""Knight (3,1) leaper — 3-axis 3-D knight."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import validate_piece_at

# 24 unique (3,1,1) permutations with all sign combinations
VECTORS_31 = [
    # (3,1,1) family
    (3, 1, 1), (3, 1, -1), (3, -1, 1), (3, -1, -1),
    (-3, 1, 1), (-3, 1, -1), (-3, -1, 1), (-3, -1, -1),
    # (1,3,1) family
    (1, 3, 1), (1, 3, -1), (1, -3, 1), (1, -3, -1),
    (-1, 3, 1), (-1, 3, -1), (-1, -3, 1), (-1, -3, -1),
    # (1,1,3) family
    (1, 1, 3), (1, 1, -3), (1, -1, 3), (1, -1, -3),
    (-1, 1, 3), (-1, 1, -3), (-1, -1, 3), (-1, -1, -3),
]

def generate_knight31_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all (3,1,1) knight leaps."""
    start = (x, y, z)
    if not validate_piece_at(state, start, pos, expected_type=PieceType.KNIGHT31):
        return []

    moves = []
    current_color = state.color

    for offset in VECTORS_31:
        target = (x + offset[0], y + offset[1], z + offset[2])
        if not (0 <= target[0] < 9 and 0 <= target[1] < 9 and 0 <= target[2] < 9):
            continue

        occupant = state.board.piece_at(target)
        if occupant is not None and occupant.color == current_color:
            continue

        is_capture = occupant is not None and occupant.color != current_color
        moves.append(Move(
            from_coord=start,
            to_coord=target,
            is_capture=is_capture
        ))

    return moves
