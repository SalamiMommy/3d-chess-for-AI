"""Counter-clockwise 3-D spiral slider — projects 6 spiral rays (one per face)."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import validate_piece_at

# Precomputed spiral offsets for each face direction
# Each entry: (direction_axis, perpendicular_plane, spiral_offsets)
# spiral_offsets: list of (step_along_axis, offset_in_plane)
_SPIRAL_PATTERNS = {
    # +X direction: axis=X, plane=YZ
    (1, 0, 0): [(1, (0, 0)), (2, (0, 1)), (3, (-1, 1)), (4, (-1, 0)), (5, (-1, -1)), (6, (0, -1)), (7, (1, -1)), (8, (1, 0))],
    # -X direction: axis=X, plane=YZ (reverse spiral)
    (-1, 0, 0): [(1, (0, 0)), (2, (0, -1)), (3, (-1, -1)), (4, (-1, 0)), (5, (-1, 1)), (6, (0, 1)), (7, (1, 1)), (8, (1, 0))],
    # +Y direction: axis=Y, plane=XZ
    (0, 1, 0): [(1, (0, 0)), (2, (-1, 0)), (3, (-1, 1)), (4, (0, 1)), (5, (1, 1)), (6, (1, 0)), (7, (1, -1)), (8, (0, -1))],
    # -Y direction: axis=Y, plane=XZ (reverse spiral)
    (0, -1, 0): [(1, (0, 0)), (2, (1, 0)), (3, (1, 1)), (4, (0, 1)), (5, (-1, 1)), (6, (-1, 0)), (7, (-1, -1)), (8, (0, -1))],
    # +Z direction: axis=Z, plane=XY
    (0, 0, 1): [(1, (0, 0)), (2, (1, 0)), (3, (1, 1)), (4, (0, 1)), (5, (-1, 1)), (6, (-1, 0)), (7, (-1, -1)), (8, (0, -1))],
    # -Z direction: axis=Z, plane=XY (reverse spiral)
    (0, 0, -1): [(1, (0, 0)), (2, (-1, 0)), (3, (-1, 1)), (4, (0, 1)), (5, (1, 1)), (6, (1, 0)), (7, (1, -1)), (8, (0, -1))],
}

def generate_spiral_moves(board, color: Color, x: int, y: int, z: int) -> List['Move']:
    """
    Generate spiral moves: 6 rays (one per face), each rotating CCW with radius 2.
    """
    start = (x, y, z)
    if not validate_piece_at(state, start, pos, expected_type=PieceType.SPIRAL):
        return []

    current_color = state.color
    board = state.board
    moves: List['Move'] = []

    for direction, spiral_steps in _SPIRAL_PATTERNS.items():
        dx, dy, dz = direction
        pos_x, pos_y, pos_z = x, y, z

        for step, (offset_a, offset_b) in spiral_steps:
            # Move along main axis
            pos_x += dx
            pos_y += dy
            pos_z += dz

            # Apply spiral offset in perpendicular plane
            if direction == (1, 0, 0) or direction == (-1, 0, 0):  # X-axis
                target = (pos_x, y + offset_a, z + offset_b)
            elif direction == (0, 1, 0) or direction == (0, -1, 0):  # Y-axis
                target = (x + offset_a, pos_y, z + offset_b)
            else:  # Z-axis
                target = (x + offset_a, y + offset_b, pos_z)

            # Bounds check
            if not (0 <= target[0] < 9 and 0 <= target[1] < 9 and 0 <= target[2] < 9):
                break

            # Stop if target is start position (shouldn't happen)
            if target == start:
                continue

            occupant = board.piece_at(target)
            if occupant is not None:
                if occupant.color != current_color:
                    moves.append(Move(from_coord=start, to_coord=target, is_capture=True))
                break  # blocked
            moves.append(Move(from_coord=start, to_coord=target, is_capture=False))

    return moves
