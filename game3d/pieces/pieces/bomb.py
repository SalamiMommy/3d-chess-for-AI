"""Bomb movement generator - king steps with self-detonation."""

import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, RADIUS_2_OFFSETS, SIZE, MOVE_FLAGS
)
from game3d.common.shared_types import Color, PieceType
from game3d.common.registry import register
from game3d.movement.movepiece import Move
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
# FIXED: Use np.any to keep rows where AT LEAST ONE coord is non-zero
origin_mask = np.any(all_coords != 0, axis=1)
BOMB_MOVEMENT_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

def generate_bomb_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:  # Changed from List[Move] to np.ndarray
    """Generate bomb moves: king-like walks + strategic self-detonation."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position
    if pos_arr.ndim == 1:
        if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
            return np.empty((0, 6), dtype=COORD_DTYPE)

    # 1. King-like movement using jump generator
    jump_engine = get_jump_movement_generator()
    moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=BOMB_MOVEMENT_VECTORS,
        allow_capture=True,
        allow_zero_direction=True,  # BOMB needs this for self-detonation
        piece_type=PieceType.KING # Use King precomputed moves for movement
    )

    # 2. Self-detonation if it would affect enemy pieces
    # Handle batch input for self-detonation check
    if pos_arr.ndim == 2:
        # Batch check
        should_detonate = _batch_has_enemy_in_explosion_range(cache_manager, pos_arr, color)
        # should_detonate is boolean array (N,)
        
        if np.any(should_detonate):
            detonating_pos = pos_arr[should_detonate]
            # Create self-destruct moves (from pos to pos)
            # Shape (K, 6) where K is number of detonating pieces
            detonate_moves = np.hstack([detonating_pos, detonating_pos]).astype(COORD_DTYPE)
            
            if moves.shape[0] > 0:
                moves = np.vstack([moves, detonate_moves])
            else:
                moves = detonate_moves
    else:
        # Single check
        if _has_enemy_in_explosion_range(cache_manager, pos_arr, color):
            # Create a self-destruct move (from pos to pos) as array
            detonate_move = np.concatenate([pos_arr, pos_arr]).reshape(1, 6).astype(COORD_DTYPE)
            # Concatenate with existing moves
            if moves.shape[0] > 0:
                moves = np.vstack([moves, detonate_move])
            else:
                moves = detonate_move

    return moves

def _batch_has_enemy_in_explosion_range(
    cache_manager: 'OptimizedCacheManager',
    centers: np.ndarray,
    current_color: int
) -> np.ndarray:
    """Check if explosion radius contains any enemy pieces for a batch of bombs."""
    # centers: (N, 3)
    # offsets: (M, 3)
    # targets: (N, M, 3)
    targets = centers[:, np.newaxis, :] + RADIUS_2_OFFSETS[np.newaxis, :, :]
    
    N, M, _ = targets.shape
    flat_targets = targets.reshape(N * M, 3)
    
    # Filter bounds
    in_bounds = in_bounds_vectorized(flat_targets)
    
    # Check occupancy for all targets
    # We need to be careful: out of bounds targets should not be checked or should return empty
    # batch_get_colors_only might handle out of bounds if we use safe version, but unsafe is faster
    # So let's use safe get or mask
    
    # Use flattened occupancy for speed
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    
    # Calculate indices
    x = flat_targets[:, 0]
    y = flat_targets[:, 1]
    z = flat_targets[:, 2]
    
    # Mask for valid indices
    valid_indices_mask = (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE) & (z >= 0) & (z < SIZE)
    
    # Initialize result array (False)
    has_enemy = np.zeros(N, dtype=np.bool_)
    
    # Only check valid targets
    if np.any(valid_indices_mask):
        valid_indices = x[valid_indices_mask] + SIZE * y[valid_indices_mask] + SIZE * SIZE * z[valid_indices_mask]
        occupants = flattened_occ[valid_indices]
        
        # Check for enemies
        is_enemy = (occupants != 0) & (occupants != current_color)
        
        # Map back to bombs
        # We need to know which bomb each target belongs to
        # flat index i corresponds to bomb i // M
        
        # Get indices of targets that have enemies
        enemy_target_indices = np.where(valid_indices_mask)[0][is_enemy]
        
        if enemy_target_indices.size > 0:
            bomb_indices = enemy_target_indices // M
            has_enemy[bomb_indices] = True
            
    return has_enemy

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
