"""Bomb movement generator - king steps with self-detonation."""

import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, RADIUS_2_OFFSETS, SIZE
)
from game3d.common.shared_types import Color, PieceType
from game3d.common.registry import register
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Bomb-specific movement vectors - king-like movement (26 directions excluding center)
# Converted to numpy-native using meshgrid for better performance
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.all(all_coords != 0, axis=1)
BOMB_MOVEMENT_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

def generate_bomb_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:  # Changed from List[Move] to np.ndarray
    """Generate bomb moves: king-like walks + strategic self-detonation."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position
    if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # 1. King-like movement using jump generator
    jump_engine = get_jump_movement_generator(cache_manager)
    moves = jump_engine.generate_jump_moves(
        color=color,
        pos=pos_arr,
        directions=BOMB_MOVEMENT_VECTORS,
        allow_capture=True,
    )

    # 2. Self-detonation if it would affect enemy pieces
    if _has_enemy_in_explosion_range(cache_manager, pos_arr, color):
        # Create a self-destruct move (from pos to pos) as array
        detonate_move = np.concatenate([pos_arr, pos_arr]).reshape(1, 6).astype(COORD_DTYPE)
        # Concatenate with existing moves
        if moves.shape[0] > 0:
            moves = np.vstack([moves, detonate_move])
        else:
            moves = detonate_move

    return moves

def _has_enemy_in_explosion_range(
    cache_manager: 'OptimizedCacheManager',
    center: np.ndarray,
    current_color: int
) -> bool:
    """Check if explosion radius contains any enemy pieces."""
    # Use precomputed radius-2 offsets from shared_types
    explosion_offsets = center + RADIUS_2_OFFSETS

    # Filter to bounds
    valid_coords = explosion_offsets[
        (explosion_offsets >= 0).all(axis=1) &
        (explosion_offsets < SIZE).all(axis=1)
    ]

    # Check each coordinate in explosion range
    for coord in valid_coords:
        victim = cache_manager.occupancy_cache.get(coord)
        if victim is not None and victim["color"] != current_color:
            return True

    return False

@register(PieceType.BOMB)
def bomb_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Generate all bomb moves for given position."""
    return generate_bomb_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_bomb_moves", "BOMB_MOVEMENT_VECTORS"]
