# game3d/movement/pieces/infiltrator.py
"""
Infiltrator – king moves + teleport to squares in front of enemy pawns.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.common.cache_utils import ensure_int_coords
from game3d.common.coord_utils import in_bounds

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# --------------------------------------------------------------------------- #
#  King directions (1-step moves)                                             #
# --------------------------------------------------------------------------- #
_KING_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

def generate_infiltrator_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate infiltrator moves: king walks + pawn-front teleports."""
    x, y, z = ensure_int_coords(x, y, z)

    # Get teleport directions
    teleport_dirs = _get_pawn_front_directions(cache, color, x, y, z)

    # Combine directions
    if len(teleport_dirs) > 0:
        all_dirs = np.unique(np.vstack((teleport_dirs, _KING_DIRECTIONS)), axis=0)
    else:
        all_dirs = _KING_DIRECTIONS

    # Generate all moves using jump movement
    jump_gen = get_integrated_jump_movement_generator(cache_manager)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=all_dirs,
        allow_capture=True,
    )

    return moves

def _get_pawn_front_directions(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> np.ndarray:
    """Get directions to empty squares in front of enemy pawns."""
    start = np.array([x, y, z], dtype=np.int16)
    enemy_color = color.opposite()

    front_squares = []

    # Find empty squares in front of enemy pawns
    for coord, piece in cache.occupancy.iter_color(enemy_color):
        if piece.ptype != PieceType.PAWN:
            continue

        # Front direction depends on enemy color
        dz = 1 if enemy_color == Color.BLACK else -1
        front = (coord[0], coord[1], coord[2] + dz)

        if in_bounds(front) and cache.occupancy.get(front) is None:
            front_squares.append(front)

    if not front_squares:
        return np.empty((0, 3), dtype=np.int8)

    targets = np.array(front_squares, dtype=np.int16)
    directions = (targets - start).astype(np.int8)

    return directions

@register(PieceType.INFILTRATOR)
def infiltrator_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_infiltrator_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_infiltrator_moves"]
