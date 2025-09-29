"""Network Teleporter — teleports to any empty square adjacent to any friendly piece."""

from typing import List, Set
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds, add_coords
from game3d.cache.manager import OptimizedCacheManager

# Precomputed 26 3D neighbor directions
_NEIGHBOR_DIRECTIONS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == dy == dz == 0)
]

def generate_network_teleport_moves(
    cache: OptimizedCacheManager,  # ← CHANGED: board → cache
    color: Color,
    x: int,
    y: int,
    z: int
) -> List['Move']:
    """
    Generate all teleport moves to empty squares adjacent to ANY friendly piece.
    Includes squares adjacent to the teleporter itself.
    """
    self_pos = (x, y, z)
    current_color = color

    # Validate piece
    piece = cache.piece_cache.get(self_pos)
    if piece is None or piece.color != current_color or piece.ptype != PieceType.FRIENDLYTELEPORTER:
        return []

    candidate_targets: Set[tuple] = set()

    # ✅ Only iterate over occupied squares (not entire board!)
    for pos, other_piece in cache.board.list_occupied():  # ← cache.board, not board
        if other_piece.color == current_color:  # friendly piece
            for dx, dy, dz in _NEIGHBOR_DIRECTIONS:
                target = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                if not in_bounds(target):
                    continue
                if cache.piece_cache.get(target) is None:  # only empty squares
                    candidate_targets.add(target)

    # Create moves
    return [
        Move(from_coord=self_pos, to_coord=target, is_capture=False)
        for target in candidate_targets
    ]
