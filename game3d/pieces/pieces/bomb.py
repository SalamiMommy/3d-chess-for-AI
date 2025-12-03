"""Bomb movement generator - king steps with self-detonation."""

import numpy as np
from typing import List, TYPE_CHECKING
from numba import njit

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, RADIUS_2_OFFSETS, SIZE, MOVE_FLAGS, SIZE_SQUARED
)
from game3d.common.shared_types import Color, PieceType
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

@njit(cache=True)
def _check_explosion_numba(
    centers: np.ndarray,
    flattened_occ: np.ndarray,
    current_color: int,
    radius_offsets: np.ndarray
) -> np.ndarray:
    """Fused kernel to check if explosion radius contains any enemy pieces."""
    n = centers.shape[0]
    n_offsets = radius_offsets.shape[0]
    has_enemy = np.zeros(n, dtype=BOOL_DTYPE)
    
    for i in range(n):
        cx, cy, cz = centers[i]
        
        for j in range(n_offsets):
            dx, dy, dz = radius_offsets[j]
            tx, ty, tz = cx + dx, cy + dy, cz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                occ = flattened_occ[idx]
                
                if occ != 0 and occ != current_color:
                    has_enemy[i] = True
                    break # Found one enemy, enough to detonate
                    
    return has_enemy

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
        # Reshape for consistent processing
        pos_arr = pos_arr.reshape(1, 3)

    # 1. King-like movement using jump generator
    jump_engine = get_jump_movement_generator()
    moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=KING_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.KING, # Use King precomputed moves for movement
        buffed_directions=BUFFED_KING_MOVEMENT_VECTORS
    )

    # 2. Self-detonation if it would affect enemy pieces
    # Use fused Numba kernel for explosion check
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    should_detonate = _check_explosion_numba(
        pos_arr, flattened_occ, color, RADIUS_2_OFFSETS
    )
    
    if np.any(should_detonate):
        detonating_pos = pos_arr[should_detonate]
        # Create self-destruct moves (from pos to pos)
        # Shape (K, 6) where K is number of detonating pieces
        detonate_moves = np.hstack([detonating_pos, detonating_pos]).astype(COORD_DTYPE)
        
        if moves.shape[0] > 0:
            moves = np.vstack([moves, detonate_moves])
        else:
            moves = detonate_moves

    return moves

@register(PieceType.BOMB)
def bomb_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Generate all bomb moves for given position."""
    return generate_bomb_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_bomb_moves"]
