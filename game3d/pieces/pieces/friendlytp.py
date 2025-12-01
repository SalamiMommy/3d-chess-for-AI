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
def _get_network_squares(
    friendly_coords: np.ndarray,
    flattened_occ: np.ndarray
) -> np.ndarray:
    """
    Fused kernel to find ALL valid teleport destinations (empty neighbors of friendly pieces).
    """
    n_friendly = friendly_coords.shape[0]
    n_dirs = _KING_DIRECTIONS.shape[0]
    
    # Use boolean mask for deduplication
    mask = np.zeros(SIZE * SIZE_SQUARED, dtype=BOOL_DTYPE)
    
    for i in range(n_friendly):
        fx, fy, fz = friendly_coords[i]
        
        for j in range(n_dirs):
            dx, dy, dz = _KING_DIRECTIONS[j]
            tx, ty, tz = fx + dx, fy + dy, fz + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                
                # Must be empty
                if flattened_occ[idx] == 0:
                    mask[idx] = True
                    
    # Collect results
    count = 0
    for i in range(SIZE * SIZE_SQUARED):
        if mask[i]:
            count += 1
            
    if count == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
        
    out = np.empty((count, 3), dtype=COORD_DTYPE)
    idx_out = 0
    
    for i in range(SIZE * SIZE_SQUARED):
        if mask[i]:
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

@njit(cache=True)
def _generate_teleport_moves_kernel(
    teleporter_positions: np.ndarray,
    network_squares: np.ndarray
) -> np.ndarray:
    """
    Generate teleport moves: Cartesian product of teleporters x network_squares.
    Excludes self-teleport.
    """
    n_tps = teleporter_positions.shape[0]
    n_targets = network_squares.shape[0]
    
    max_moves = n_tps * n_targets
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    
    count = 0
    for i in range(n_tps):
        sx, sy, sz = teleporter_positions[i]
        
        for j in range(n_targets):
            tx, ty, tz = network_squares[j]
            
            # Skip self-teleport
            if sx == tx and sy == ty and sz == tz:
                continue
                
            moves[count, 0] = sx
            moves[count, 1] = sy
            moves[count, 2] = sz
            moves[count, 3] = tx
            moves[count, 4] = ty
            moves[count, 5] = tz
            count += 1
            
    return moves[:count]

def generate_friendlytp_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate friendly teleporter moves: king walks + network teleports."""
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Handle single input by reshaping
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)
    
    moves_list = []
    
    # 1. King walks (using jump engine)
    jump_engine = get_jump_movement_generator()
    king_moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=_KING_DIRECTIONS,
        allow_capture=True,
        piece_type=PieceType.FRIENDLYTELEPORTER
    )
    if king_moves.size > 0:
        moves_list.append(king_moves)
        
    # 2. Network Teleports
    # Get all friendly pieces
    friendly_coords = cache_manager.occupancy_cache.get_positions(color)
    flattened_occ = cache_manager.occupancy_cache.get_flattened_occupancy()
    
    # Find all valid teleport destinations (empty neighbors of friendly pieces)
    network_squares = _get_network_squares(friendly_coords, flattened_occ)
    
    if network_squares.shape[0] > 0:
        # Generate moves for all teleporters to all network squares
        teleport_moves = _generate_teleport_moves_kernel(pos_arr, network_squares)
        if teleport_moves.size > 0:
            moves_list.append(teleport_moves)
            
    if not moves_list:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    return np.concatenate(moves_list, axis=0)

@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_friendlytp_moves(state.cache_manager, state.color, pos)

__all__ = ["_KING_DIRECTIONS", "generate_friendlytp_moves"]
