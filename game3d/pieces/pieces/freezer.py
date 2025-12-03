"""Freezer piece - king-like movement with freeze aura."""
from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from numba import njit

from game3d.common.shared_types import (
    COORD_DTYPE, RADIUS_2_OFFSETS, Color, PieceType, SIZE, SIZE_SQUARED, BOOL_DTYPE
)
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

def generate_freezer_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    pos_arr = pos.astype(COORD_DTYPE)

    # Use jump generator with piece-specific vectors
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=KING_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.FREEZER,
        buffed_directions=BUFFED_KING_MOVEMENT_VECTORS
    )

@njit(cache=True)
def _get_frozen_squares_fast(
    freezer_positions: np.ndarray,
    flattened_occ: np.ndarray,
    friendly_color: int,
    radius_offsets: np.ndarray
) -> np.ndarray:
    """Fused kernel to find enemy pieces within radius 2 of freezers.
    
    Replaces:
    1. Expansion (freezer + offsets)
    2. Bounds checking
    3. Occupancy lookup
    4. Enemy filtering
    5. Deduplication (via boolean mask)
    """
    # Use a boolean map to track unique squares
    # Since board is small (9x9x9 = 729), a boolean map is efficient.
    affected_mask = np.zeros(SIZE_SQUARED * SIZE, dtype=BOOL_DTYPE)
    
    n_freezers = freezer_positions.shape[0]
    n_offsets = radius_offsets.shape[0]
    
    for i in range(n_freezers):
        sx, sy, sz = freezer_positions[i]
        
        for j in range(n_offsets):
            dx, dy, dz = radius_offsets[j]
            tx, ty, tz = sx + dx, sy + dy, sz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                occ = flattened_occ[idx]
                
                # Check for enemy piece (not empty, not friendly)
                if occ != 0 and occ != friendly_color:
                    affected_mask[idx] = True
                    
    # Count unique affected squares
    count = 0
    for i in range(SIZE_SQUARED * SIZE):
        if affected_mask[i]:
            count += 1
            
    if count == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
            
    # Collect coordinates
    out = np.empty((count, 3), dtype=COORD_DTYPE)
    idx_out = 0
    for i in range(SIZE_SQUARED * SIZE):
        if affected_mask[i]:
            # Decode index
            z = i // SIZE_SQUARED
            rem = i % SIZE_SQUARED
            y = rem // SIZE
            x = rem % SIZE
            out[idx_out, 0] = x
            out[idx_out, 1] = y
            out[idx_out, 2] = z
            idx_out += 1
            
    return out

def get_all_frozen_squares_numpy(
    cache_manager: 'OptimizedCacheManager',
    controller: Color,
) -> np.ndarray:
    """Get all enemy squares frozen by controller's freezers. Returns (N, 3) array."""
    if isinstance(controller, np.ndarray):
        controller = Color(int(controller.item()))
    
    # Get all friendly pieces
    all_coords = cache_manager.occupancy_cache.get_positions(controller)
    if all_coords.size == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
    
    # Filter for freezers
    # Use unsafe batch get for speed since coords are from get_positions
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(all_coords)
    freezer_mask = piece_types == PieceType.FREEZER
    freezers = all_coords[freezer_mask]

    if freezers.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    # Use fused Numba kernel
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    return _get_frozen_squares_fast(
        freezers, flattened_occ, int(controller), RADIUS_2_OFFSETS
    )

@register(PieceType.FREEZER)
def freezer_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Generate legal freezer moves from position."""
    return generate_freezer_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_freezer_moves", "get_all_frozen_squares_numpy"]
