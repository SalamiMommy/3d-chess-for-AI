"""Knight (3,2) leaper — 3-axis 3-D knight."""

from typing import List
from game3d.pieces.enums import PieceType, Color
from game3d.movement.pathvalidation import validate_piece_at
from game3d.movement.movepiece import Move  # ← Import Move
from game3d.cache.manager import OptimizedCacheManager
# 48 unique (3,2,1) permutations with all sign combinations
VECTORS_32 = [
    # (3,2,1) family
    (3, 2, 1), (3, 2, -1), (3, -2, 1), (3, -2, -1),
    (-3, 2, 1), (-3, 2, -1), (-3, -2, 1), (-3, -2, -1),
    # (3,1,2) family
    (3, 1, 2), (3, 1, -2), (3, -1, 2), (3, -1, -2),
    (-3, 1, 2), (-3, 1, -2), (-3, -1, 2), (-3, -1, -2),
    # (2,3,1) family
    (2, 3, 1), (2, 3, -1), (2, -3, 1), (2, -3, -1),
    (-2, 3, 1), (-2, 3, -1), (-2, -3, 1), (-2, -3, -1),
    # (1,3,2) family
    (1, 3, 2), (1, 3, -2), (1, -3, 2), (1, -3, -2),
    (-1, 3, 2), (-1, 3, -2), (-1, -3, 2), (-1, -3, -2),
    # (2,1,3) family
    (2, 1, 3), (2, 1, -3), (2, -1, 3), (2, -1, -3),
    (-2, 1, 3), (-2, 1, -3), (-2, -1, 3), (-2, -1, -3),
    # (1,2,3) family
    (1, 2, 3), (1, 2, -3), (1, -2, 3), (1, -2, -3),
    (-1, 2, 3), (-1, 2, -3), (-1, -2, 3), (-1, -2, -3),
]

def generate_knight32_moves(cache: OptimizedCacheManager, color: Color, x: int, y: int, z: int) -> List['Move']:
    """Generate all (3,2,1) knight leaps."""
    start = (x, y, z)

    # Validate piece at start position
    if not validate_piece_at(cache, color, start, expected_type=PieceType.KNIGHT32):
        return []

    moves = []

    for dx, dy, dz in VECTORS_32:
        target = (x + dx, y + dy, z + dz)
        # Bounds check (BOARD_SIZE = 9)
        if not (0 <= target[0] < 9 and 0 <= target[1] < 9 and 0 <= target[2] < 9):
            continue

        occupant = cache.piece_cache.get(target)
        if occupant is not None and occupant.color == color:
            continue  # Skip friendly pieces

        is_capture = occupant is not None and occupant.color != color
        moves.append(Move(
            from_coord=start,
            to_coord=target,
            is_capture=is_capture
        ))

    return moves
