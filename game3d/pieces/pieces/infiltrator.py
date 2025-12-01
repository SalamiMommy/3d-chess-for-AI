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

# King directions (1-step moves) - converted to numpy-native using meshgrid
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.any(all_coords != 0, axis=1)
_KING_DIRECTIONS = all_coords[origin_mask].astype(COORD_DTYPE)

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
    """Generate infiltrator moves: king walks + pawn-front teleports."""
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
        directions=_KING_DIRECTIONS,
        allow_capture=True,
        piece_type=PieceType.INFILTRATOR
    )
    if king_moves.size > 0:
        moves_list.append(king_moves)
        
    # 2. Pawn-front teleports
    # Get all valid pawn front squares
    pawn_fronts = _get_valid_pawn_fronts(cache_manager, color)
    
    if pawn_fronts.shape[0] > 0:
        teleport_moves = _generate_infiltrator_teleport_moves(start, pawn_fronts)
        if teleport_moves.size > 0:
            moves_list.append(teleport_moves)
            
    if not moves_list:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    return np.concatenate(moves_list, axis=0)

def _get_valid_pawn_fronts(
    cache_manager: 'OptimizedCacheManager',
    color: int
) -> np.ndarray:
    """Get all empty squares in front of enemy pawns."""
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

    # Front direction depends on enemy color
    dz = 1 if enemy_color == Color.BLACK else -1
    # Vectorized addition
    front_squares = pawn_coords.copy()
    front_squares[:, 2] += dz

    # Check bounds
    valid_mask = in_bounds_vectorized(front_squares)
    valid_front_squares = front_squares[valid_mask]

    if valid_front_squares.shape[0] == 0:
        return get_empty_coord_batch()

    # Filter to empty squares
    empty_mask = ~cache_manager.occupancy_cache.batch_is_occupied_unsafe(valid_front_squares)
    empty_front_squares = valid_front_squares[empty_mask]

    return empty_front_squares

@register(PieceType.INFILTRATOR)
def infiltrator_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_infiltrator_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_infiltrator_moves"]
