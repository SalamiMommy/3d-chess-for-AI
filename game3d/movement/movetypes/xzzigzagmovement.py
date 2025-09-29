"""XZ-Zig-Zag Slider — zig-zag rays along coordinate planes."""

from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.pathvalidation import validate_piece_at
from game3d.cache.manager import OptimizedCacheManager

SEGMENT_LENGTH = 3

def _zigzag_ray(
    cache: OptimizedCacheManager,  # ← CHANGED: board → cache
    color: Color,
    start: tuple[int, int, int],
    plane: str,      # 'XZ', 'XY', 'YZ'
    primary_dir: int,    # +1 or -1 for first axis
    secondary_dir: int,  # +1 or -1 for second axis
    fixed_coord: int
) -> List['Move']:
    """Generate one zig-zag ray in the specified plane."""
    x, y, z = start
    current_color = color
    moves: List['Move'] = []

    # Determine which coordinates are variable
    if plane == 'XZ':  # Y fixed
        var1, var2 = x, z
        axis1, axis2 = 'x', 'z'
    elif plane == 'XY':  # Z fixed
        var1, var2 = x, y
        axis1, axis2 = 'x', 'y'
    else:  # 'YZ' - X fixed
        var1, var2 = y, z
        axis1, axis2 = 'y', 'z'

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
            if plane == 'XZ':
                target = (curr1, fixed_coord, curr2)
            elif plane == 'XY':
                target = (curr1, curr2, fixed_coord)
            else:  # YZ
                target = (fixed_coord, curr1, curr2)

            # Bounds check
            if not (0 <= target[0] < 9 and 0 <= target[1] < 9 and 0 <= target[2] < 9):
                return moves

            occupant = cache.piece_cache.get(target)  # ← NOW cache is defined
            if occupant is not None:
                if occupant.color != current_color:
                    moves.append(Move(from_coord=start, to_coord=target, is_capture=True))
                return moves

            moves.append(Move(from_coord=start, to_coord=target, is_capture=False))
            steps_taken += 1

        # Switch axis for next segment
        move_axis_1 = not move_axis_1

def generate_xz_zigzag_moves(
    cache: OptimizedCacheManager,  # ← CHANGED: board → cache
    color: Color,
    x: int,
    y: int,
    z: int
) -> List['Move']:
    """Generate all zig-zag moves in XZ, XY, and YZ planes."""
    start = (x, y, z)
    if not validate_piece_at(cache, color, start, PieceType.XZZIGZAG):  # ← cache, not board
        return []

    moves: List['Move'] = []
    directions = [(1, -1), (-1, 1)]

    # XZ plane (Y fixed)
    for pri, sec in directions:
        moves.extend(_zigzag_ray(cache, color, start, 'XZ', pri, sec, y))  # ← cache, not board

    # XY plane (Z fixed)
    for pri, sec in directions:
        moves.extend(_zigzag_ray(cache, color, start, 'XY', pri, sec, z))  # ← cache, not board

    # YZ plane (X fixed)
    for pri, sec in directions:
        moves.extend(_zigzag_ray(cache, color, start, 'YZ', pri, sec, x))  # ← cache, not board

    return moves
