# game3d/movement/pieces/mirror.py - Mirror piece implementation
"""
Mirror-Teleporter: Teleports to mirrored position across board center.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE_MINUS_1
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def generate_mirror_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate mirror teleport moves to mirrored position."""
    start_coord = pos.astype(COORD_DTYPE)
    x, y, z = start_coord[0], start_coord[1], start_coord[2]

    # Calculate mirrored target position
    target_coord = np.array([SIZE_MINUS_1 - x, SIZE_MINUS_1 - y, SIZE_MINUS_1 - z], dtype=COORD_DTYPE)

    # Verify target validity
    if np.array_equal(start_coord, target_coord) or not in_bounds_vectorized(target_coord.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Check if target has friendly piece
    colors, _ = cache_manager.occupancy_cache.batch_get_attributes(target_coord.reshape(1, 3))
    if colors[0] != 0 and colors[0] == (1 if color == Color.WHITE else 2):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Calculate direction for jump movement
    direction = target_coord - start_coord

    # Generate move using jump movement
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start_coord,
        directions=direction.reshape(1, 3),
        allow_capture=True,
        piece_type=PieceType.MIRROR
    )

@register(PieceType.MIRROR)
def mirror_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatch mirror piece moves."""
    return generate_mirror_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_mirror_moves"]
