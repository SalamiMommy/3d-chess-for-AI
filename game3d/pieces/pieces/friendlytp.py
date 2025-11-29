# game3d/movement/pieces/friendlytp.py - OPTIMIZED NUMBA VERSION
"""
Friendly-Teleporter – teleport to any empty neighbour of a friendly piece
PLUS normal 1-step King moves.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch
from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, BOOL_DTYPE
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# King directions (1-step moves)
# Pre-computed as constant for Numba
_KING_DIRECTIONS = np.array([
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0],
    [1, 0, 0], [-1, 0, 0]
], dtype=COORD_DTYPE)

@njit(cache=True, fastmath=True)
def _build_network_directions_numba(
    start: np.ndarray,
    friendly_coords: np.ndarray,
    flattened_occ: np.ndarray
) -> np.ndarray:
    """
    Fused kernel to find teleport directions.
    Generates neighbors of all friendly pieces, filters bounds/occupancy,
    deduplicates, and calculates direction vectors from start.
    """
    n_friendly = friendly_coords.shape[0]
    n_dirs = _KING_DIRECTIONS.shape[0]
    
    # Use boolean mask for deduplication
    mask = np.zeros(SIZE * SIZE_SQUARED, dtype=BOOL_DTYPE)
    
    # Start index to exclude self-teleport
    start_idx = start[0] + SIZE * start[1] + SIZE_SQUARED * start[2]
    
    for i in range(n_friendly):
        fx, fy, fz = friendly_coords[i]
        
        # Skip if friendly piece is the teleporter itself (optional, but cleaner)
        if fx == start[0] and fy == start[1] and fz == start[2]:
            continue
            
        for j in range(n_dirs):
            dx, dy, dz = _KING_DIRECTIONS[j]
            tx, ty, tz = fx + dx, fy + dy, fz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                
                # Must be empty and not the start square
                if idx != start_idx and flattened_occ[idx] == 0:
                    mask[idx] = True
                    
    # Collect results
    count = 0
    for i in range(SIZE * SIZE_SQUARED):
        if mask[i]:
            count += 1
            
    if count == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
        
    # We return DIRECTIONS from start, not absolute coords
    out_dirs = np.empty((count, 3), dtype=COORD_DTYPE)
    idx_out = 0
    
    for i in range(SIZE * SIZE_SQUARED):
        if mask[i]:
            # Decode index
            z = i // SIZE_SQUARED
            rem = i % SIZE_SQUARED
            y = rem // SIZE
            x = rem % SIZE
            
            # Calculate direction vector
            out_dirs[idx_out, 0] = x - start[0]
            out_dirs[idx_out, 1] = y - start[1]
            out_dirs[idx_out, 2] = z - start[2]
            idx_out += 1
            
    return out_dirs

def generate_friendlytp_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate friendly teleporter moves: king walks + network teleports."""
    pos_arr = pos.astype(COORD_DTYPE)
    
    jump_engine = get_jump_movement_generator()
    
    # Handle batch input
    if pos_arr.ndim == 2:
        moves_list = []
        friendly_coords = cache_manager.occupancy_cache.get_positions(color)
        flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
        
        for i in range(pos_arr.shape[0]):
            start = pos_arr[i]
            teleport_dirs = _build_network_directions_numba(start, friendly_coords, flattened_occ)
            
            if teleport_dirs.shape[0] > 0:
                all_dirs = np.vstack((_KING_DIRECTIONS, teleport_dirs))
            else:
                all_dirs = _KING_DIRECTIONS
                
            moves = jump_engine.generate_jump_moves(
                cache_manager=cache_manager,
                color=color,
                pos=start,
                directions=all_dirs,
                allow_capture=True,
                piece_type=PieceType.FRIENDLYTELEPORTER
            )
            if moves.shape[0] > 0:
                moves_list.append(moves)
                
        if not moves_list:
            return np.empty((0, 6), dtype=COORD_DTYPE)
        return np.concatenate(moves_list, axis=0)

    # Single input path
    start = pos_arr

    # Get all friendly pieces
    friendly_coords = cache_manager.occupancy_cache.get_positions(color)
    
    # Build network directions using fused kernel
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    teleport_dirs = _build_network_directions_numba(start, friendly_coords, flattened_occ)

    # Combine with normal King moves
    if teleport_dirs.shape[0] > 0:
        all_dirs = np.vstack((_KING_DIRECTIONS, teleport_dirs))
    else:
        all_dirs = _KING_DIRECTIONS

    # Generate all moves
    moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=all_dirs,
        allow_capture=True,
        piece_type=PieceType.FRIENDLYTELEPORTER
    )

    return moves

@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_friendlytp_moves(state.cache_manager, state.color, pos)

__all__ = ["_KING_DIRECTIONS", "generate_friendlytp_moves"]
