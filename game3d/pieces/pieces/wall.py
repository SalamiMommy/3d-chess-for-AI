# game3d/movement/movetypes/wall.py - OPTIMIZED BATCH NUMBA VERSION
"""
Unified Wall movement + behind-capture logic - optimized with Numba.
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np
from numba import njit, prange

from game3d.common.shared_types import Color, PieceType, Result, WALL, COORD_DTYPE, SIZE, get_empty_coord_batch, BOOL_DTYPE
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds_vectorized
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

# Wall block offsets (constant)
WALL_BLOCK_OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=COORD_DTYPE)

@njit(cache=True, fastmath=True)
def _wall_squares_numba(anchor: np.ndarray) -> np.ndarray:
    """Return the 4 squares occupied by a Wall anchored at anchor."""
    # (4, 3)
    res = np.empty((4, 3), dtype=COORD_DTYPE)
    for i in range(4):
        res[i, 0] = anchor[0] + WALL_BLOCK_OFFSETS[i, 0]
        res[i, 1] = anchor[1] + WALL_BLOCK_OFFSETS[i, 1]
        res[i, 2] = anchor[2] + WALL_BLOCK_OFFSETS[i, 2]
    return res

@njit(cache=True, fastmath=True)
def _block_in_bounds_numba(anchor: np.ndarray) -> bool:
    """True if the entire 2×2×1 block stays inside the board."""
    # Anchor must be strictly less than SIZE - 1 to allow room for the +1 block
    return (0 <= anchor[0] < SIZE - 1 and 
            0 <= anchor[1] < SIZE - 1 and 
            0 <= anchor[2] < SIZE)

@njit(cache=True, fastmath=True)
def _build_behind_mask_numba(anchor: np.ndarray) -> np.ndarray:
    """
    Return every square that is **behind** the Wall anchored at *anchor*.
    Returns array of shape (N, 3) containing all behind squares.
    """
    # Max possible squares: 6 directions * (SIZE-1) steps * 4 squares/block?
    # No, "behind" means along the movement lines from the wall.
    # The original implementation generated rays from the anchor.
    # Let's replicate that logic but faster.
    
    # Original logic: anchor + direction * step for all 6 directions
    # Max squares = 6 * (SIZE-1)
    max_squares = 6 * (SIZE - 1)
    out = np.empty((max_squares, 3), dtype=COORD_DTYPE)
    count = 0
    
    for d in range(6):
        dx, dy, dz = KING_MOVEMENT_VECTORS[d]
        
        # Start from 1 step away
        curr_x = anchor[0] + dx
        curr_y = anchor[1] + dy
        curr_z = anchor[2] + dz
        
        while 0 <= curr_x < SIZE and 0 <= curr_y < SIZE and 0 <= curr_z < SIZE:
            out[count, 0] = curr_x
            out[count, 1] = curr_y
            out[count, 2] = curr_z
            count += 1
            
            curr_x += dx
            curr_y += dy
            curr_z += dz
            
    return out[:count]

@njit(cache=True, fastmath=True, parallel=True)
def _can_capture_wall_numba(attacker_sqs: np.ndarray, wall_anchors: np.ndarray) -> np.ndarray:
    """
    Check if attackers can capture walls.
    Returns boolean array of shape (num_attackers,)
    """
    n_attackers = attacker_sqs.shape[0]
    n_walls = wall_anchors.shape[0]
    result = np.zeros(n_attackers, dtype=BOOL_DTYPE)
    
    if n_walls == 0:
        return result
        
    for i in prange(n_attackers):
        ax, ay, az = attacker_sqs[i]
        
        for j in range(n_walls):
            wx, wy, wz = wall_anchors[j]
            
            # Check alignment
            dx = ax - wx
            dy = ay - wy
            dz = az - wz
            
            # Count zeros
            zeros = 0
            if dx == 0: zeros += 1
            if dy == 0: zeros += 1
            if dz == 0: zeros += 1
            
            if zeros == 2:
                # Aligned on one axis. Check direction.
                # Must be "behind" -> diff > 0?
                # Original logic: diff > 0
                
                is_valid = False
                if dx != 0:
                    if dx > 0: is_valid = True
                elif dy != 0:
                    if dy > 0: is_valid = True
                elif dz != 0:
                    if dz > 0: is_valid = True
                    
                if is_valid:
                    result[i] = True
                    break # Found a wall this attacker can capture
                    
    return result

def can_capture_wall_vectorized(attacker_sqs: np.ndarray, wall_anchors: np.ndarray) -> np.ndarray:
    """Public API wrapper."""
    if attacker_sqs.shape[0] == 0:
        return np.array([], dtype=bool)
    return _can_capture_wall_numba(
        attacker_sqs.astype(COORD_DTYPE), 
        wall_anchors.astype(COORD_DTYPE)
    )

def is_wall_anchor(pos: np.ndarray, cache_manager: 'OptimizedCacheManager') -> bool:
    """Check if pos is the top-left anchor of a 2x2 wall block."""
    x, y, z = pos[0], pos[1], pos[2]
    
    # Check left neighbor (x-1)
    if x > 0:
        if hasattr(cache_manager.occupancy_cache, 'get_type_at'):
            type_l = cache_manager.occupancy_cache.get_type_at(x-1, y, z)
        else:
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

@njit(cache=True, fastmath=True)
def _generate_wall_moves_batch_kernel(
    anchors: np.ndarray, 
    occ: np.ndarray,
    my_color: int,
    board_size: int,
    directions: np.ndarray
) -> np.ndarray:
    """
    Fused kernel for wall move generation for a batch of walls.
    Input:
        anchors: (N, 3)
        occ: (SIZE, SIZE, SIZE) - occupancy grid
        my_color: int
        board_size: int - board dimension (usually 9)
        directions: (D, 3) - movement vectors
    Returns:
        (M, 6) moves
    """
    n_walls = anchors.shape[0]
    n_dirs = directions.shape[0]
    max_moves = n_walls * n_dirs
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    count = 0
    
    # Current wall squares for overlap check (reusable buffer)
    current_squares = np.empty((4, 3), dtype=COORD_DTYPE)
    
    for w in range(n_walls):
        anchor = anchors[w]
        
        # Bounds check for anchor itself (must be valid anchor)
        if not (0 <= anchor[0] < board_size - 1 and 
                0 <= anchor[1] < board_size - 1 and 
                0 <= anchor[2] < board_size):
            continue
            
        # ✅ OPTIMIZED: Precompute current squares as flat indices for O(1) lookup
        # Instead of 4 coordinate comparisons per target square (16 total), use flat index
        current_flat_indices = np.empty(4, dtype=np.int64)
        for k in range(4):
            cx = anchor[0] + WALL_BLOCK_OFFSETS[k, 0]
            cy = anchor[1] + WALL_BLOCK_OFFSETS[k, 1]
            cz = anchor[2] + WALL_BLOCK_OFFSETS[k, 2]
            current_flat_indices[k] = cx + cy * board_size + cz * board_size * board_size
        
        for i in range(n_dirs):
            # Target anchor for this direction
            tax = anchor[0] + directions[i, 0]
            tay = anchor[1] + directions[i, 1]
            taz = anchor[2] + directions[i, 2]
            
            # Bounds check for target anchor
            if tax < 0 or tax > board_size - 2: continue
            if tay < 0 or tay > board_size - 2: continue
            if taz < 0 or taz > board_size - 1: continue
                
            # Check the 4 squares of the target wall
            is_blocked = False
            
            for j in range(4):
                # Target square coords
                tx = tax + WALL_BLOCK_OFFSETS[j, 0]
                ty = tay + WALL_BLOCK_OFFSETS[j, 1]
                tz = taz + WALL_BLOCK_OFFSETS[j, 2]
                
                # Bounds check before array access
                if not (0 <= tx < board_size and 0 <= ty < board_size and 0 <= tz < board_size):
                    is_blocked = True
                    break
                
                # Direct occupancy lookup
                occ_color = occ[tx, ty, tz]
                
                if occ_color != 0:
                    # Square is occupied. Check if it's a friendly piece that isn't part of current wall
                    if occ_color == my_color:
                        # ✅ OPTIMIZED: O(1) flat index comparison instead of 4x coordinate loop
                        target_flat = tx + ty * board_size + tz * board_size * board_size
                        is_self = False
                        for k in range(4):
                            if current_flat_indices[k] == target_flat:
                                is_self = True
                                break
                        
                        if not is_self:
                            is_blocked = True
                            break
            
            if not is_blocked:
                moves[count, 0] = anchor[0]
                moves[count, 1] = anchor[1]
                moves[count, 2] = anchor[2]
                moves[count, 3] = tax
                moves[count, 4] = tay
                moves[count, 5] = taz
                count += 1
            
    return moves[:count]



@njit(cache=True, fastmath=True)
def _filter_anchors_numba(
    anchors: np.ndarray,
    ptype: np.ndarray
) -> np.ndarray:
    """
    Filter anchors to ensure they are the top-left of a wall.
    Returns indices of valid anchors.
    """
    n = anchors.shape[0]
    valid_indices = np.empty(n, dtype=np.int32)
    count = 0
    
    for i in range(n):
        x, y, z = anchors[i]
        
        # Check left neighbor (x-1)
        is_left_wall = False
        if x > 0:
            if ptype[x-1, y, z] == PieceType.WALL:
                is_left_wall = True
                
        # Check up neighbor (y-1)
        is_up_wall = False
        if y > 0:
            if ptype[x, y-1, z] == PieceType.WALL:
                is_up_wall = True
                
        if not is_left_wall and not is_up_wall:
            valid_indices[count] = i
            count += 1
            
    return valid_indices[:count]

def generate_wall_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """
    Generate Wall moves with fused buffed/unbuffed processing.
    
    ✅ OPTIMIZED: 
    - Trusted kernel-level bounds validation (no redundant Python checks)
    - Fused buffed/unbuffed into single kernel call
    - Filter anchors with fast Numba kernel
    """
    anchors = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if anchors.ndim == 1:
        anchors = anchors.reshape(1, 3)
        
    if anchors.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Filter out non-anchor wall parts using Numba kernel
    # This also validates anchor bounds within the kernel
    valid_indices = _filter_anchors_numba(anchors, cache_manager.occupancy_cache._ptype)
    
    if valid_indices.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    valid_anchors = anchors[valid_indices]
    
    if valid_anchors.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    # ✅ OPTIMIZED: Removed Python-level bounds validation - kernel handles this
    # Get buffed status for all anchors at once
    buffed_squares = cache_manager.consolidated_aura_cache._buffed_squares
    is_buffed = buffed_squares[valid_anchors[:, 0], valid_anchors[:, 1], valid_anchors[:, 2]]
    
    # Define orthogonal vectors for unbuffed wall (6 directions)
    UNBUFFED_WALL_VECTORS = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=COORD_DTYPE)
    
    # Call fused kernel (handles all bounds checking internally)
    return _generate_wall_moves_fused_kernel(
        valid_anchors,
        cache_manager.occupancy_cache._occ,
        color,
        SIZE,
        UNBUFFED_WALL_VECTORS,
        BUFFED_KING_MOVEMENT_VECTORS,
        is_buffed.astype(np.int8)
    )

@njit(cache=True, fastmath=True)
def _generate_wall_moves_fused_kernel(
    anchors: np.ndarray, 
    occ: np.ndarray,
    my_color: int,
    board_size: int,
    unbuffed_directions: np.ndarray,
    buffed_directions: np.ndarray,
    is_buffed: np.ndarray
) -> np.ndarray:
    """
    Fused kernel for wall move generation handling both buffed and unbuffed walls.
    
    ✅ OPTIMIZED: Single pass with conditional direction selection.
    """
    n_walls = anchors.shape[0]
    # Max moves = max dirs (buffed) per wall
    max_dirs = max(unbuffed_directions.shape[0], buffed_directions.shape[0])
    max_moves = n_walls * max_dirs
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    count = 0
    
    # Current wall squares for overlap check (reusable buffer)
    current_squares = np.empty((4, 3), dtype=COORD_DTYPE)
    
    for w in range(n_walls):
        anchor = anchors[w]
        
        # Select directions based on buffed status
        if is_buffed[w]:
            directions = buffed_directions
        else:
            directions = unbuffed_directions
        
        n_dirs = directions.shape[0]
        
        # ✅ OPTIMIZED: Precompute current squares as flat indices for O(1) lookup
        current_flat_indices = np.empty(4, dtype=np.int64)
        for k in range(4):
            cx = anchor[0] + WALL_BLOCK_OFFSETS[k, 0]
            cy = anchor[1] + WALL_BLOCK_OFFSETS[k, 1]
            cz = anchor[2] + WALL_BLOCK_OFFSETS[k, 2]
            current_flat_indices[k] = cx + cy * board_size + cz * board_size * board_size
        
        for i in range(n_dirs):
            # Target anchor for this direction
            tax = anchor[0] + directions[i, 0]
            tay = anchor[1] + directions[i, 1]
            taz = anchor[2] + directions[i, 2]
            
            # Bounds check for target anchor
            if tax < 0 or tax > board_size - 2: continue
            if tay < 0 or tay > board_size - 2: continue
            if taz < 0 or taz > board_size - 1: continue
                
            # Check the 4 squares of the target wall
            is_blocked = False
            
            for j in range(4):
                tx = tax + WALL_BLOCK_OFFSETS[j, 0]
                ty = tay + WALL_BLOCK_OFFSETS[j, 1]
                tz = taz + WALL_BLOCK_OFFSETS[j, 2]
                
                # Bounds check before array access
                if not (0 <= tx < board_size and 0 <= ty < board_size and 0 <= tz < board_size):
                    is_blocked = True
                    break
                
                # Direct occupancy lookup
                occ_color = occ[tx, ty, tz]
                
                if occ_color != 0:
                    if occ_color == my_color:
                        # ✅ OPTIMIZED: O(1) flat index comparison
                        target_flat = tx + ty * board_size + tz * board_size * board_size
                        is_self = False
                        for k in range(4):
                            if current_flat_indices[k] == target_flat:
                                is_self = True
                                break
                        
                        if not is_self:
                            is_blocked = True
                            break
            
            if not is_blocked:
                moves[count, 0] = anchor[0]
                moves[count, 1] = anchor[1]
                moves[count, 2] = anchor[2]
                moves[count, 3] = tax
                moves[count, 4] = tay
                moves[count, 5] = taz
                count += 1
            
    return moves[:count]

@register(PieceType.WALL)
def wall_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_wall_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_wall_moves", "can_capture_wall_vectorized"]

