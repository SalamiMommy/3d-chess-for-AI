"""YZ-Zig-Zag Slider — zig-zag rays along coordinate planes."""

from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import validate_piece_at

SEGMENT_LENGTH = 3

def _zigzag_ray(
    board,
    color: Color,
    start: tuple[int, int, int],
    plane: str,      # 'YZ', 'XZ', 'XY'
    primary_dir: int,    # +1 or -1 for first axis
    secondary_dir: int,  # +1 or -1 for second axis
    fixed_coord: int
) -> List['Move']:
    """Generate one zig-zag ray in the specified plane."""
    x, y, z = start
    current_color = color
    moves: List['Move'] = []

    # Determine variable coordinates based on plane
    if plane == 'YZ':  # X fixed
        var1, var2 = y, z
    elif plane == 'XZ':  # Y fixed
        var1, var2 = x, z
    else:  # 'XY' - Z fixed
        var1, var2 = x, y

    # Track current position
    curr1, curr2 = var1, var2
    move_axis_1 = True  # Start by moving along first axis

    while True:
        direction = primary_dir if move_axis_1 else secondary_dir
        steps_taken = 0

        while steps_taken < SEGMENT_LENGTH:
            # Move along current axis
            if move_axis_1:
                curr1 += direction
            else:
                curr2 += direction

            # Build target coordinate
            if plane == 'YZ':
                target = (fixed_coord, curr1, curr2)
            elif plane == 'XZ':
                target = (curr1, fixed_coord, curr2)
            else:  # XY
                target = (curr1, curr2, fixed_coord)

            # Bounds check
            if not (0 <= target[0] < 9 and 0 <= target[1] < 9 and 0 <= target[2] < 9):
                return moves

            occupant = cache.piece_cache.get(target)
            if occupant is not None:
                if occupant.color != current_color:
                    moves.append(Move(from_coord=start, to_coord=target, is_capture=True))
                return moves

            moves.append(Move(from_coord=start, to_coord=target, is_capture=False))
            steps_taken += 1

        # Switch axis for next segment
        move_axis_1 = not move_axis_1

def generate_yz_zigzag_moves(
    board,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate all zig-zag moves in YZ, XZ, and XY planes."""
    start = (x, y, z)
    if not validate_piece_at(board, color, start, PieceType.YZZIGZAG):
        return []

    moves: List['Move'] = []
    directions = [(1, -1), (-1, 1)]

    # YZ plane (X fixed)
    for pri, sec in directions:
        moves.extend(_zigzag_ray(board, color, start, 'YZ', pri, sec, x))

    # XZ plane (Y fixed)
    for pri, sec in directions:
        moves.extend(_zigzag_ray(board, color, start, 'XZ', pri, sec, y))

    # XY plane (Z fixed)
    for pri, sec in directions:
        moves.extend(_zigzag_ray(board, color, start, 'XY', pri, sec, z))

    return moves
