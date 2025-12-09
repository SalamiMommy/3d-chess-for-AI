# game3d/movement/pieces/infiltrator.py - FULLY NUMPY-NATIVE
"""
Infiltrator – king moves + teleport to squares in front of enemy pawns.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch
from game3d.common.shared_types import *
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

@njit(cache=True)
def _generate_infiltrator_teleport_moves(
    infiltrators: np.ndarray,
    pawn_fronts: np.ndarray
) -> np.ndarray:
    """
    Generate teleport moves from all infiltrators to all valid pawn fronts.
    Returns array of moves (N, 6).
    """
    n_inf = infiltrators.shape[0]
    n_targets = pawn_fronts.shape[0]
    
    max_moves = n_inf * n_targets
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    
    count = 0
    for i in range(n_inf):
        sx, sy, sz = infiltrators[i]
        
        for j in range(n_targets):
            tx, ty, tz = pawn_fronts[j]
            
            # Skip self-teleport (unlikely but possible if pawn front is where infiltrator is)
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

    """Generate infiltrator moves: king walks + pawn teleports (front unbuffed, behind buffed)."""
    start = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if start.ndim == 1:
        start = start.reshape(1, 3)
        
    moves_list = []
    
    # 1. King walks (using jump engine)
    jump_engine = get_jump_movement_generator()
    king_moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=KING_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.INFILTRATOR,
        buffed_directions=BUFFED_KING_MOVEMENT_VECTORS
    )
    if king_moves.size > 0:
        moves_list.append(king_moves)
        
    # 2. Pawn teleports
    # Check if buffed to determine if we teleport behind or in front
    buffed_squares = cache_manager.consolidated_aura_cache._buffed_squares
    x, y, z = start[0]
    is_buffed = buffed_squares[x, y, z]
    
    # Get valid pawn squares (front if unbuffed, behind if buffed)
    pawn_targets = _get_valid_pawn_targets(cache_manager, color, teleport_behind=is_buffed)
    
    if pawn_targets.shape[0] > 0:
        teleport_moves = _generate_infiltrator_teleport_moves(start, pawn_targets)
        if teleport_moves.size > 0:
            moves_list.append(teleport_moves)
            
    if not moves_list:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    return np.concatenate(moves_list, axis=0)

@njit(cache=True, fastmath=True)
def _get_pawn_targets_kernel(
    occ: np.ndarray,
    ptype: np.ndarray,
    enemy_color: int,
    dz: int
) -> np.ndarray:
    """
    Fused kernel to find valid pawn targets.
    Iterates over the board to find enemy pawns and checks their targets.
    """
    # Max possible targets = number of squares (upper bound)
    # We can use a smaller buffer if needed, but 729 is small.
    targets = np.empty((SIZE * SIZE * SIZE, 3), dtype=COORD_DTYPE)
    count = 0
    
    # Iterate over all squares
    # We can iterate linearly for speed
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                # Check if enemy pawn
                if occ[x, y, z] == enemy_color and ptype[x, y, z] == PieceType.PAWN:
                    # Calculate target z
                    tz = z + dz
                    
                    # Check bounds
                    if 0 <= tz < SIZE:
                        # Check if target is empty
                        if occ[x, y, tz] == 0:
                            targets[count, 0] = x
                            targets[count, 1] = y
                            targets[count, 2] = tz
                            count += 1
                            
    return targets[:count]

def _get_valid_pawn_targets(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    teleport_behind: bool = False
):
    """Get empty squares in front of (or behind) enemy pawns using fused kernel."""
    enemy_color = Color(color).opposite().value
    
    # Direction depends on enemy color and whether we want front or behind
    # If enemy is Black, they move -Z, so front is -1, behind is +1
    # If enemy is White, they move +Z, so front is +1, behind is -1
    if teleport_behind:
        # Behind is opposite of pawn's forward direction
        dz = 1 if enemy_color == Color.BLACK else -1
    else:
        # Front is same as pawn's forward direction
        dz = -1 if enemy_color == Color.BLACK else 1
        
    return _get_pawn_targets_kernel(
        cache_manager.occupancy_cache._occ,
        cache_manager.occupancy_cache._ptype,
        enemy_color,
        dz
    )

__all__ = []

