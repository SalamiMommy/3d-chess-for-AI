# game3d/movement/movetypes/wall.py - FULLY NUMPY-NATIVE
"""
Unified Wall movement + behind-capture logic - fully vectorized.
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import Color, PieceType, Result, WALL, COORD_DTYPE, SIZE, get_empty_coord_batch
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# Wall-specific movement vectors - orthogonal movement (6 directions) - numpy native
WALL_MOVEMENT_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

# Wall block offsets (constant)
WALL_BLOCK_OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=COORD_DTYPE)

# 2×2×1 block geometry helpers
def _wall_squares_numpy(anchor: np.ndarray) -> np.ndarray:
    """Return the 4 squares occupied by a Wall anchored at anchor."""
    return anchor.reshape(1, 3) + WALL_BLOCK_OFFSETS

def _block_in_bounds_numpy(anchor: np.ndarray) -> bool:
    """True if the entire 2×2×1 block stays inside the board."""
    # Check anchor and anchor + [1, 1, 0]
    # Anchor must be strictly less than SIZE - 1 to allow room for the +1 block
    if not (0 <= anchor[0] < SIZE - 1 and 0 <= anchor[1] < SIZE - 1 and 0 <= anchor[2] < SIZE):
        return False
    return True

# Behind-mask builder (fully vectorized)
def _build_behind_mask_numpy(anchor: np.ndarray) -> np.ndarray:
    """
    Return every square that is **behind** the Wall anchored at *anchor*.
    Returns array of shape (N, 3) containing all behind squares.
    Fully vectorized implementation.
    """
    directions = WALL_MOVEMENT_VECTORS

    # Vectorized calculation of all possible positions in each direction
    max_steps = SIZE - 1
    step_range = np.arange(1, max_steps + 1, dtype=COORD_DTYPE)

    # Calculate all positions: anchor + direction * step for all combinations
    direction_expanded = directions[:, np.newaxis, :]
    step_expanded = step_range[np.newaxis, :, np.newaxis]
    anchor_expanded = anchor[np.newaxis, np.newaxis, :]

    all_positions = anchor_expanded + direction_expanded * step_expanded

    # Flatten and filter valid positions
    positions_2d = all_positions.reshape(-1, 3)

    # Vectorized bounds checking
    valid_mask = np.all((positions_2d >= 0) & (positions_2d < SIZE), axis=1)
    valid_positions = positions_2d[valid_mask]

    if valid_positions.shape[0] == 0:
        return get_empty_coord_batch()

    return valid_positions.astype(COORD_DTYPE)

def can_capture_wall_vectorized(attacker_sqs: np.ndarray, wall_anchors: np.ndarray) -> np.ndarray:
    """Check if attackers can capture walls - fully vectorized numpy operations."""
    if attacker_sqs.shape[0] == 0:
        return np.array([], dtype=bool)

    attacker_sqs = np.asarray(attacker_sqs, dtype=COORD_DTYPE)
    wall_anchors = np.asarray(wall_anchors, dtype=COORD_DTYPE)

    # Broadcast for all combinations: (num_attackers, num_walls, 3)
    diff = attacker_sqs[:, np.newaxis, :] - wall_anchors[np.newaxis, :, :]

    # Check alignment in each axis (2 coordinates must be zero, 1 must be non-zero)
    zero_mask = (diff == 0)
    zero_count = np.sum(zero_mask, axis=2)
    aligned_mask = (zero_count == 2)

    # For aligned pairs, check direction validity
    non_zero_mask = ~zero_mask
    non_zero_axis = np.where(non_zero_mask)

    if len(non_zero_axis[0]) > 0:
        attacker_indices, wall_indices, axis_indices = non_zero_axis
        diff_values = diff[attacker_indices, wall_indices, axis_indices]
        valid_direction_mask = diff_values > 0
        valid_directions = np.zeros_like(aligned_mask)
        valid_directions[attacker_indices, wall_indices] = valid_direction_mask

        return (aligned_mask & valid_directions).any(axis=1)

    return np.zeros(attacker_sqs.shape[0], dtype=bool)

def can_capture_wall_numpy(attacker_sq: np.ndarray, wall_anchor: np.ndarray) -> bool:
    """
    Return True if *attacker_sq* is **behind** the Wall anchored at *wall_anchor*.
    Numpy native version.
    """
    behind_mask = _build_behind_mask_numpy(wall_anchor)
    if len(behind_mask) == 0:
        return False

    # Check if attacker_sq matches any behind square
    return np.any(np.all(behind_mask == attacker_sq.reshape(1, 3), axis=1))

def is_wall_anchor(pos: np.ndarray, cache_manager: 'OptimizedCacheManager') -> bool:
    """Check if pos is the top-left anchor of a 2x2 wall block.
    
    OPTIMIZED: Uses scalar accessors to avoid array creation.
    """
    x, y, z = pos[0], pos[1], pos[2]
    
    # Check left neighbor (x-1)
    if x > 0:
        # Check if left neighbor is a WALL
        # Use scalar accessor if available (it is now)
        if hasattr(cache_manager.occupancy_cache, 'get_type_at'):
            type_l = cache_manager.occupancy_cache.get_type_at(x-1, y, z)
        else:
            # Fallback (should not happen with updated cache)
            left = np.array([x-1, y, z], dtype=COORD_DTYPE)
            type_l, _ = cache_manager.occupancy_cache.get_fast(left)
            
        if type_l == PieceType.WALL:
            return False
            
    # Check up neighbor (y-1)
    if y > 0:
        if hasattr(cache_manager.occupancy_cache, 'get_type_at'):
            type_u = cache_manager.occupancy_cache.get_type_at(x, y-1, z)
        else:
            up = np.array([x, y-1, z], dtype=COORD_DTYPE)
            type_u, _ = cache_manager.occupancy_cache.get_fast(up)
            
        if type_u == PieceType.WALL:
            return False
            
    return True

def generate_wall_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    anchor = pos.astype(COORD_DTYPE)

    # 1. Enforce Anchor Logic: Only the top-left piece generates moves
    if not is_wall_anchor(anchor, cache_manager):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    if not _block_in_bounds_numpy(anchor):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # 2. Vectorized Move Generation
    # Calculate all 6 potential target anchors at once
    # WALL_MOVEMENT_VECTORS shape is (6, 3)
    target_anchors = anchor + WALL_MOVEMENT_VECTORS
    
    # Filter out-of-bounds target anchors
    # For a 2x2 block at anchor (x,y,z), we need:
    # 0 <= x < SIZE - 1
    # 0 <= y < SIZE - 1
    # 0 <= z < SIZE
    valid_mask = (
        (target_anchors[:, 0] >= 0) & (target_anchors[:, 0] < SIZE - 1) &
        (target_anchors[:, 1] >= 0) & (target_anchors[:, 1] < SIZE - 1) &
        (target_anchors[:, 2] >= 0) & (target_anchors[:, 2] < SIZE)
    )
    
    valid_target_anchors = target_anchors[valid_mask]
    
    if valid_target_anchors.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    # 3. Batch Occupancy Check
    # We need to check 4 squares for each valid target anchor
    # Total squares to check = num_valid_moves * 4
    
    # Offsets for the 4 squares of the wall: (0,0,0), (1,0,0), (0,1,0), (1,1,0)
    # Use constant to avoid allocation
    block_offsets = WALL_BLOCK_OFFSETS
    
    # Create all target squares: (num_moves, 4, 3)
    # Broadcasting: (num_moves, 1, 3) + (1, 4, 3) -> (num_moves, 4, 3)
    all_target_squares = valid_target_anchors[:, np.newaxis, :] + block_offsets[np.newaxis, :, :]
    
    # Flatten to (num_moves * 4, 3) for batch lookup
    flat_target_squares = all_target_squares.reshape(-1, 3)
    
    # Batch lookup attributes
    # Use unsafe variant as coordinates are guaranteed valid by valid_mask logic
    colors, types = cache_manager.occupancy_cache.batch_get_attributes_unsafe(flat_target_squares)
    
    # Reshape back to (num_moves, 4)
    colors = colors.reshape(-1, 4)
    types = types.reshape(-1, 4)
    
    # 4. Vectorized Validity Logic
    # Valid if:
    # - Square is empty (color == 0)
    # - Square is occupied by opponent (Capture)
    # - Square is occupied by self, BUT it's part of the current wall (Self-overlap allowed)
    
    # Check for self-overlap
    # Current wall squares: anchor + block_offsets
    current_wall_squares = anchor + block_offsets # (4, 3)
    
    # We need to check if each target square matches ANY of the current wall squares
    # all_target_squares: (num_moves, 4, 3)
    # current_wall_squares: (4, 3)
    # Result: (num_moves, 4) boolean mask where True means "is part of current wall"
    
    # Expand dims for broadcasting:
    # (num_moves, 4, 1, 3) == (1, 1, 4, 3) -> (num_moves, 4, 4, 3)
    # Check equality on last axis (coords), then any on the 3rd axis (matching any current square)
    
    # Optimization: We know the geometry.
    # A move by (dx, dy, dz) overlaps if:
    # - dx=0, dy=0, dz!=0: All 4 squares move to new Z. No overlap (unless dz=0 which is impossible)
    # - dx=1: (0,0)->(1,0), (0,1)->(1,1). 2 squares overlap.
    # - dx=-1: (1,0)->(0,0), (1,1)->(0,1). 2 squares overlap.
    # - dy=1: (0,0)->(0,1), (1,0)->(1,1). 2 squares overlap.
    # - dy=-1: (0,1)->(0,0), (1,1)->(1,0). 2 squares overlap.
    
    # Let's use the general approach for correctness, it's vectorized anyway.
    # Actually, we can just check if target_square is in current_wall_squares.
    # But doing this for every square is expensive (N*4*4 comparisons).
    
    # Faster approach:
    # Identify friendly pieces (colors == color)
    friendly_mask = (colors == color)
    
    # If no friendly pieces, we don't need to check overlap
    if not np.any(friendly_mask):
        # All non-empty are opponents -> captures
        # Blocked if ANY square is friendly (already checked) or if we can't capture?
        # Wait, wall captures if it moves into opponent.
        # It is blocked if it moves into friendly (unless self-overlap).
        pass
        
    # Self-overlap check
    # A square is "self" if it equals any of the current wall squares
    # We can pre-calculate this based on the move direction, but let's be robust.
    # We can use the fact that we are moving the ANCHOR.
    # target_sq = target_anchor + offset
    # current_sq = anchor + offset'
    # target_sq == current_sq  =>  target_anchor + offset == anchor + offset'
    # => move_vec + offset == offset'
    # => move_vec == offset' - offset
    
    # This is still complex. Let's use the coordinate comparison.
    # (num_moves, 4, 3) vs (4, 3)
    # We can use isin or similar, but manual broadcasting is fine for small 4x4.
    
    diffs = all_target_squares[:, :, np.newaxis, :] - current_wall_squares[np.newaxis, np.newaxis, :, :]
    # (num_moves, 4, 4, 3)
    
    # Check where diff is 0,0,0
    is_self_overlap = np.any(np.all(diffs == 0, axis=3), axis=2) # (num_moves, 4)
    
    # Logic:
    # A square blocks if:
    # - It is occupied (color != 0)
    # - AND it is NOT self-overlap
    # - AND (it is friendly OR (it is opponent AND cannot be captured?))
    # Wait, Wall captures ANY opponent. So opponent never blocks (unless special rules apply like Armour?)
    # Assuming Wall crushes everything.
    
    # So, blocked if:
    # - Friendly AND NOT self-overlap
    
    is_friendly = (colors == color)
    is_blocking_friendly = is_friendly & (~is_self_overlap)
    
    # Move is blocked if ANY of its 4 squares are blocking friendly
    move_is_blocked = np.any(is_blocking_friendly, axis=1)
    
    # Filter valid moves
    valid_indices = np.where(~move_is_blocked)[0]
    
    if valid_indices.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    final_moves = valid_target_anchors[valid_indices]
    
    # Construct result: (N, 6) [from_x, from_y, from_z, to_x, to_y, to_z]
    # We use the ANCHOR as the from/to coordinates
    n_moves = final_moves.shape[0]
    result = np.empty((n_moves, 6), dtype=COORD_DTYPE)
    result[:, :3] = anchor
    result[:, 3:] = final_moves
    
    return result

@register(PieceType.WALL)
def wall_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_wall_moves(state.cache_manager, state.color, pos)

__all__ = ["WALL_MOVEMENT_VECTORS", "generate_wall_moves", "can_capture_wall_numpy", "can_capture_wall_vectorized"]
