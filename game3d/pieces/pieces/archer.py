# game3d/movement/pieces/archer.py - FULLY NUMPY-NATIVE
"""
Unified Archer dispatcher
- 1-radius sphere  → walk (normal king-like move)
- 2-radius surface → shoot (archery capture, no movement)
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import TYPE_CHECKING
from game3d.common.shared_types import *
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

# Archery directions (2-radius surface only) - optimized numpy construction
coords = np.mgrid[-2:3, -2:3, -2:3].reshape(3, -1).T
distances = np.sum(coords * coords, axis=1)
_ARCHERY_DIRECTIONS = coords[distances == 4].astype(COORD_DTYPE)

@njit(cache=True, fastmath=True)
def _generate_archer_shots_kernel(
    starts: np.ndarray,
    directions: np.ndarray,
    flattened_occ: np.ndarray,
    color: int
) -> np.ndarray:
    """
    Numba kernel to generate archer shots.
    Filters bounds and enemy occupancy inline to avoid large intermediate arrays.
    """
    n_starts = starts.shape[0]
    n_dirs = directions.shape[0]
    
    # Pre-calculate max possible moves to allocate once
    max_moves = n_starts * n_dirs
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    
    count = 0
    
    for i in range(n_starts):
        sx, sy, sz = starts[i]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j]
            tx, ty, tz = sx + dx, sy + dy, sz + dz
            
            # Bounds check
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                # Occupancy check
                idx = tx + SIZE * ty + SIZE_SQUARED * tz
                occ = flattened_occ[idx]
                
                # Must be enemy
                if occ != 0 and occ != color:
                    moves[count, 0] = sx
                    moves[count, 1] = sy
                    moves[count, 2] = sz
                    moves[count, 3] = tx
                    moves[count, 4] = ty
                    moves[count, 5] = tz
                    count += 1
                    
    return moves[:count]

def generate_archer_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> list[Move]:
    """Generate all archer moves: king walks + archery shots."""
    start = pos.astype(COORD_DTYPE)

    moves_list = []

    # 1. King walks using jump movement (already vectorized)
    jump_gen = get_jump_movement_generator()
    king_moves = jump_gen.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=KING_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.ARCHER,
        buffed_directions=BUFFED_KING_MOVEMENT_VECTORS
    )
    if king_moves.size > 0:
        moves_list.append(king_moves)

    # 2. Archery shots (2-radius surface capture only) - NUMBA KERNEL
    
    # Handle batch input for archery shots
    if start.ndim == 1:
        start = start.reshape(1, 3)
        
    flattened = cache_manager.occupancy_cache.get_flattened_occupancy()
    
    shot_moves = _generate_archer_shots_kernel(
        start,
        _ARCHERY_DIRECTIONS,
        flattened,
        color
    )
    
    if shot_moves.size > 0:
        moves_list.append(shot_moves)

    if not moves_list:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    return np.concatenate(moves_list, axis=0)

@register(PieceType.ARCHER)
def archer_move_dispatcher(state: 'GameState', pos: np.ndarray) -> list[Move]:
    return generate_archer_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_archer_moves"]
