"""Counter-clockwise 3-D spiral slider — projects 6 spiral rays (one per face)."""

from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import validate_piece_at
from game3d.movement.movepiece import Move  # Ensure Move is imported!

# Precomputed spiral offsets per face direction
_SPIRAL_PATTERNS = {
    (1, 0, 0):  [(1, (0, 0)), (2, (0, 1)), (3, (-1, 1)), (4, (-1, 0)), (5, (-1, -1)), (6, (0, -1)), (7, (1, -1)), (8, (1, 0))],
    (-1, 0, 0): [(1, (0, 0)), (2, (0, -1)), (3, (-1, -1)), (4, (-1, 0)), (5, (-1, 1)), (6, (0, 1)), (7, (1, 1)), (8, (1, 0))],
    (0, 1, 0):  [(1, (0, 0)), (2, (-1, 0)), (3, (-1, 1)), (4, (0, 1)), (5, (1, 1)), (6, (1, 0)), (7, (1, -1)), (8, (0, -1))],
    (0, -1, 0): [(1, (0, 0)), (2, (1, 0)), (3, (1, 1)), (4, (0, 1)), (5, (-1, 1)), (6, (-1, 0)), (7, (-1, -1)), (8, (0, -1))],
    (0, 0, 1):  [(1, (0, 0)), (2, (1, 0)), (3, (1, 1)), (4, (0, 1)), (5, (-1, 1)), (6, (-1, 0)), (7, (-1, -1)), (8, (0, -1))],
    (0, 0, -1): [(1, (0, 0)), (2, (-1, 0)), (3, (-1, 1)), (4, (0, 1)), (5, (1, 1)), (6, (1, 0)), (7, (1, -1)), (8, (0, -1))],
}

def generate_spiral_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate spiral moves: 6 rays (one per face), each rotating CCW with radius 2."""
    start = (x, y, z)

    # Validate piece at start
    if not validate_piece_at(board, color, start, expected_type=PieceType.SPIRAL):
        return []

    moves: List[Move] = []

    for direction, spiral_steps in _SPIRAL_PATTERNS.items():
        dx, dy, dz = direction

        for step, (offset_a, offset_b) in spiral_steps:
            # Position along main axis: start + step * direction
            along_x = x + step * dx
            along_y = y + step * dy
            along_z = z + step * dz

            # Apply spiral offset in perpendicular plane
            if dx != 0:  # X-axis movement → offset in YZ plane
                target = (along_x, along_y + offset_a, along_z + offset_b)
            elif dy != 0:  # Y-axis → offset in XZ plane
                target = (along_x + offset_a, along_y, along_z + offset_b)
            else:  # Z-axis → offset in XY plane
                target = (along_x + offset_a, along_y + offset_b, along_z)

            # Bounds check (BOARD_SIZE = 9)
            if not (0 <= target[0] < 9 and 0 <= target[1] < 9 and 0 <= target[2] < 9):
                break  # Ray ends at board edge

            if target == start:
                continue  # Skip self (shouldn't occur)

            occupant = cache.piece_cache.get(target)
            if occupant is None:
                moves.append(Move(from_coord=start, to_coord=target, is_capture=False))
            elif occupant.color != color:
                moves.append(Move(from_coord=start, to_coord=target, is_capture=True))
                break  # Can't move through enemy
            else:
                break  # Blocked by friendly piece

    return moves
