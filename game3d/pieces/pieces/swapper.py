# swapper.py - FULLY NUMPY-NATIVE
"""
Swapper == King-steps ∪ friendly-swap teleport
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from numba import njit

from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch, SWAPPER, COORD_DTYPE, SIZE
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

@njit(cache=True)
def _generate_swap_moves_kernel(
    swapper_positions: np.ndarray,
    friendly_positions: np.ndarray,
    friendly_types: np.ndarray
) -> np.ndarray:
    """Fused kernel to generate swap moves (swapper -> friendly piece)."""
    n_swappers = swapper_positions.shape[0]
    n_friendly = friendly_positions.shape[0]
    
    # Max moves = n_swappers * (n_friendly - 1)
    # But we allocate safely
    max_moves = n_swappers * n_friendly
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    
    count = 0
    for i in range(n_swappers):
        sx, sy, sz = swapper_positions[i]
        
        for j in range(n_friendly):
            fx, fy, fz = friendly_positions[j]
            
            # Skip self (cannot swap with self)
            if sx == fx and sy == fy and sz == fz:
                continue
            
            # ✅ CRITICAL FIX: Prevent swapping with Wall if Swapper's position is invalid for Wall
            # Wall requires 2x2 space (x < SIZE-1, y < SIZE-1)
            if friendly_types[j] == PieceType.WALL:
                if not (sx < SIZE - 1 and sy < SIZE - 1):
                    continue
                
            moves[count, 0] = sx
            moves[count, 1] = sy
            moves[count, 2] = sz
            moves[count, 3] = fx
            moves[count, 4] = fy
            moves[count, 5] = fz
            count += 1
            
    return moves[:count]

def generate_swapper_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    jump_engine = get_jump_movement_generator()
    moves_list = []
    
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Handle single input by reshaping to (1, 3)
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)

    # 1. King walks
    # jump_engine handles batch input natively
    king_moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=KING_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.SWAPPER,
        buffed_directions=BUFFED_KING_MOVEMENT_VECTORS
    )
    if king_moves.size > 0:
        moves_list.append(king_moves)

    # 2. Friendly swaps
    # Get all friendly pieces
    friendly_coords = cache_manager.occupancy_cache.get_positions(color)
    
    if friendly_coords.shape[0] > 0:
        # Get types to check for Walls
        friendly_types = cache_manager.occupancy_cache.batch_get_types_only(friendly_coords)
        
        # Use fused kernel to generate swap moves
        swap_moves = _generate_swap_moves_kernel(pos_arr, friendly_coords, friendly_types)
        
        if swap_moves.size > 0:
            moves_list.append(swap_moves)

    if not moves_list:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    return np.concatenate(moves_list, axis=0)

@register(PieceType.SWAPPER)
def swapper_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_swapper_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_swapper_moves"]
