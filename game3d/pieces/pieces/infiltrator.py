# game3d/movement/pieces/infiltrator.py - FULLY NUMPY-NATIVE
"""
Infiltrator – king moves + teleport to squares in front of enemy pawns.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import Color, PieceType, Result, get_empty_coord_batch
from game3d.common.shared_types import (
    COORD_DTYPE, INDEX_DTYPE, BOOL_DTYPE, COLOR_DTYPE, PIECE_TYPE_DTYPE, SIZE
)
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

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

def generate_infiltrator_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
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

def _get_valid_pawn_targets(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    teleport_behind: bool = False
) -> np.ndarray:
    """Get empty squares in front of (or behind) enemy pawns.
    
    Args:
        cache_manager: Cache manager
        color: Infiltrator's color
        teleport_behind: If True, get squares behind pawns. If False, get squares in front.
    """
    enemy_color = Color(color).opposite()

    # Get all enemy piece positions
    enemy_coords = cache_manager.occupancy_cache.get_positions(enemy_color)

    if enemy_coords.shape[0] == 0:
        return get_empty_coord_batch()

    # Get piece types (unsafe is fine as coords are valid)
    _, piece_types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(enemy_coords)

    # Filter to only pawn coordinates
    pawn_mask = piece_types == PieceType.PAWN.value
    pawn_coords = enemy_coords[pawn_mask]

    if pawn_coords.shape[0] == 0:
        return get_empty_coord_batch()

    # Direction depends on enemy color and whether we want front or behind
    # If enemy is Black, they move -Z, so front is -1, behind is +1
    # If enemy is White, they move +Z, so front is +1, behind is -1
    if teleport_behind:
        # Behind is opposite of pawn's forward direction
        dz = 1 if enemy_color == Color.BLACK else -1
    else:
        # Front is same as pawn's forward direction
        dz = -1 if enemy_color == Color.BLACK else 1
        
    # Vectorized addition
    target_squares = pawn_coords.copy()
    target_squares[:, 2] += dz

    # Check bounds
    valid_mask = in_bounds_vectorized(target_squares)
    valid_target_squares = target_squares[valid_mask]

    if valid_target_squares.shape[0] == 0:
        return get_empty_coord_batch()

    # Filter to empty squares
    empty_mask = ~cache_manager.occupancy_cache.batch_is_occupied_unsafe(valid_target_squares)
    empty_target_squares = valid_target_squares[empty_mask]

    return empty_target_squares

@register(PieceType.INFILTRATOR)
def infiltrator_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_infiltrator_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_infiltrator_moves"]
