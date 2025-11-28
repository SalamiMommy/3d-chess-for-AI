"""Pawn movement generator - fully numpy native with vectorized operations."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.coord_utils import in_bounds_vectorized
from game3d.common.shared_types import (
    Color, PieceType,
    COORD_DTYPE, COLOR_WHITE, COLOR_BLACK, SIZE,
    PAWN_START_RANK_WHITE, PAWN_START_RANK_BLACK,
    PAWN_PROMOTION_RANK_WHITE, PAWN_PROMOTION_RANK_BLACK, MOVE_FLAGS
)
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Pawn push directions - white moves +Y, black moves -Y
PAWN_PUSH_DIRECTIONS = np.array([
    [0, 1, 0],   # White pawn push
    [0, -1, 0],  # Black pawn push
], dtype=COORD_DTYPE)

# Pawn attack directions - 4 trigonal attacks
# White (+Y): (±1, 1, ±1)
# Black (-Y): (±1, -1, ±1)
PAWN_ATTACK_DIRECTIONS = np.array([
    [1, 1, 1], [-1, 1, 1], [1, 1, -1], [-1, 1, -1],  # White attacks
    [1, -1, 1], [-1, -1, 1], [1, -1, -1], [-1, -1, -1],  # Black attacks
], dtype=COORD_DTYPE)

def _is_armoured(piece) -> bool:
    """Check if piece has armour protection."""
    return piece is not None and (
        (hasattr(piece, "armoured") and piece.armoured) or
        piece["piece_type"] == PieceType.ARMOUR
    )

def _is_on_start_rank(y: int, colour: Color) -> bool:
    """Check if pawn is on starting rank (using Y coordinate)."""
    return (colour == Color.WHITE and y == PAWN_START_RANK_WHITE) or \
           (colour == Color.BLACK and y == PAWN_START_RANK_BLACK)

def _is_promotion_rank(y: int, colour: Color) -> bool:
    """Check if pawn is on promotion rank (using Y coordinate)."""
    return (colour == Color.WHITE and y == PAWN_PROMOTION_RANK_WHITE) or \
           (colour == Color.BLACK and y == PAWN_PROMOTION_RANK_BLACK)

def generate_pawn_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate pawn moves with optimized fast-path for common cases."""
    start = pos.astype(COORD_DTYPE)
    x, y, z = start[0], start[1], start[2]
    colour = Color(color)

    # Validate position
    if not in_bounds_vectorized(start.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    move_arrays = []
    
    # Select appropriate push direction based on color
    color_idx = 0 if colour == Color.WHITE else 1
    dy = 1 if colour == Color.WHITE else -1
    
    # ✅ FAST PATH: Batch check both single and double push in one call
    push_y = y + dy
    two_step_y = y + 2 * dy
    
    # Check if positions are in bounds first
    single_push_valid = 0 <= push_y < SIZE
    double_push_possible = _is_on_start_rank(y, colour) and 0 <= two_step_y < SIZE
    
    if single_push_valid:
        if double_push_possible:
            # Batch check both positions at once
            check_positions = np.array([[x, push_y, z], [x, two_step_y, z]], dtype=COORD_DTYPE)
            colors, _ = cache_manager.occupancy_cache.batch_get_attributes(check_positions)
            
            if colors[0] == 0:  # push_pos empty
                # Create single push move
                single_push = np.array([[x, y, z, x, push_y, z]], dtype=COORD_DTYPE)
                move_arrays.append(single_push)
                
                if colors[1] == 0:  # two_step also empty
                    two_step_move = np.array([[x, y, z, x, two_step_y, z]], dtype=COORD_DTYPE)
                    move_arrays.append(two_step_move)
        else:
            # Only check single push
            push_pos = np.array([[x, push_y, z]], dtype=COORD_DTYPE)
            colors, _ = cache_manager.occupancy_cache.batch_get_attributes(push_pos)
            
            if colors[0] == 0:
                single_push = np.array([[x, y, z, x, push_y, z]], dtype=COORD_DTYPE)
                move_arrays.append(single_push)

    # ✅ OPTIMIZED: Batch all capture checks at once
    # Select appropriate attack directions based on color
    if color == COLOR_WHITE:
        attack_dirs = PAWN_ATTACK_DIRECTIONS[:4]  # First 4 are white attacks
    else:
        attack_dirs = PAWN_ATTACK_DIRECTIONS[4:]  # Last 4 are black attacks
    
    # Calculate all potential capture destinations
    capture_dests = start + attack_dirs
    
    # Filter for in-bounds
    in_bounds_mask = ((capture_dests[:, 0] >= 0) & (capture_dests[:, 0] < SIZE) &
                      (capture_dests[:, 1] >= 0) & (capture_dests[:, 1] < SIZE) &
                      (capture_dests[:, 2] >= 0) & (capture_dests[:, 2] < SIZE))
    
    valid_capture_dests = capture_dests[in_bounds_mask]
    
    if valid_capture_dests.shape[0] > 0:
        # ✅ OPTIMIZATION: First check colors only (faster - no type array allocation)
        dest_colors = cache_manager.occupancy_cache.batch_get_colors_only(valid_capture_dests)
        
        # Valid captures: enemy pieces
        enemy_mask = (dest_colors != 0) & (dest_colors != color)
        
        if np.any(enemy_mask):
            # Only fetch types for enemy squares (sparse check for ARMOUR)
            enemy_dests = valid_capture_dests[enemy_mask]
            _, dest_types = cache_manager.occupancy_cache.batch_get_attributes(enemy_dests)
            
            # Filter out ARMOUR
            not_armour_mask = (dest_types != PieceType.ARMOUR)
            
            if np.any(not_armour_mask):
                final_dests = enemy_dests[not_armour_mask]
                n_captures = final_dests.shape[0]
                
                # Create capture moves
                cap_moves = np.empty((n_captures, 6), dtype=COORD_DTYPE)
                cap_moves[:, :3] = start  # from coords
                cap_moves[:, 3:] = final_dests  # to coords
                
                move_arrays.append(cap_moves)

    if not move_arrays:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    return np.concatenate(move_arrays)

@register(PieceType.PAWN)
def pawn_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_pawn_moves(state.cache_manager, state.color, pos)

__all__ = ['generate_pawn_moves', 'PAWN_PUSH_DIRECTIONS', 'PAWN_ATTACK_DIRECTIONS']
