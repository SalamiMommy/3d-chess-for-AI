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
    start = pos.astype(COORD_DTYPE)
    x, y, z = start[0], start[1], start[2]
    colour = Color(color)

    # Validate position
    if not in_bounds_vectorized(start.reshape(1, 3))[0]:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    jump_engine = get_jump_movement_generator()
    move_arrays = []

    # Select appropriate push direction based on color (0=white, 1=black)
    color_idx = 0 if colour == Color.WHITE else 1
    push_dir = PAWN_PUSH_DIRECTIONS[color_idx:color_idx+1]  # Shape (1, 3)

    # Push moves
    push_moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=push_dir,
        allow_capture=False,
    )
    
    if push_moves.size > 0:
        move_arrays.append(push_moves)

    # Two-step push from start rank
    if _is_on_start_rank(y, colour):
        dy = 1 if colour == Color.WHITE else -1
        two_step_dir = np.array([[0, 2 * dy, 0]], dtype=COORD_DTYPE)
        two_step_pos = start + two_step_dir[0]

        if (in_bounds_vectorized(two_step_pos.reshape(1, 3))[0] and
            cache_manager.occupancy_cache.get(two_step_pos) is None):
            two_step_moves = jump_engine.generate_jump_moves(
                cache_manager=cache_manager,
        color=color,
                pos=start,
                directions=two_step_dir,
                allow_capture=False,
            )
            if two_step_moves.size > 0:
                move_arrays.append(two_step_moves)

    # Select appropriate attack directions based on color
    if color == COLOR_WHITE:
        attack_dirs = PAWN_ATTACK_DIRECTIONS[:4]  # First 4 are white attacks
    else:
        attack_dirs = PAWN_ATTACK_DIRECTIONS[4:]  # Last 4 are black attacks

    # Capture moves
    cap_moves = jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=attack_dirs,
        allow_capture=True,
    )

    # Filter invalid captures (armoured, etc)
    if cap_moves.size > 0:
        # We need to check victims for armour
        # cap_moves is (N, 6), destination is columns 3:6
        destinations = cap_moves[:, 3:6]
        
        # Get victims efficiently
        dest_colors, dest_types = cache_manager.occupancy_cache.batch_get_attributes(destinations)
        
        # Check for ARMOUR type (PieceType.ARMOUR = 28)
        valid_cap_mask = (dest_types != PieceType.ARMOUR)
        
        if np.all(valid_cap_mask):
            move_arrays.append(cap_moves)
        else:
            move_arrays.append(cap_moves[valid_cap_mask])

    if not move_arrays:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    return np.concatenate(move_arrays)

@register(PieceType.PAWN)
def pawn_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_pawn_moves(state.cache_manager, state.color, pos)

__all__ = ['generate_pawn_moves', 'PAWN_PUSH_DIRECTIONS', 'PAWN_ATTACK_DIRECTIONS']
