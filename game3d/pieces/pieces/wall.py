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

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# Wall-specific movement vectors - orthogonal movement (6 directions)
WALL_MOVEMENT_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

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
        dx, dy, dz = WALL_MOVEMENT_VECTORS[d]
        
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
    my_color: int
) -> np.ndarray:
    """
    Fused kernel for wall move generation for a batch of walls.
    Input:
        anchors: (N, 3)
        occ: (SIZE, SIZE, SIZE) - occupancy grid
        my_color: int
    Returns:
        (M, 6) moves
    """
    n_walls = anchors.shape[0]
    max_moves = n_walls * 6
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    count = 0
    
    # Current wall squares for overlap check (reusable buffer)
    current_squares = np.empty((4, 3), dtype=COORD_DTYPE)
    
    for w in range(n_walls):
        anchor = anchors[w]
        
        # Bounds check for anchor itself (must be valid anchor)
        if not (0 <= anchor[0] < SIZE - 1 and 
                0 <= anchor[1] < SIZE - 1 and 
                0 <= anchor[2] < SIZE):
            continue
            
        # Compute current squares for this wall
        for k in range(4):
            current_squares[k, 0] = anchor[0] + WALL_BLOCK_OFFSETS[k, 0]
            current_squares[k, 1] = anchor[1] + WALL_BLOCK_OFFSETS[k, 1]
            current_squares[k, 2] = anchor[2] + WALL_BLOCK_OFFSETS[k, 2]
        
        for i in range(6):
            # Target anchor for this direction
            tax = anchor[0] + WALL_MOVEMENT_VECTORS[i, 0]
            tay = anchor[1] + WALL_MOVEMENT_VECTORS[i, 1]
            taz = anchor[2] + WALL_MOVEMENT_VECTORS[i, 2]
            
            # Bounds check for target anchor
            if not (0 <= tax < SIZE - 1 and 0 <= tay < SIZE - 1 and 0 <= taz < SIZE):
                continue
                
            # Check the 4 squares of the target wall
            is_blocked = False
            
            for j in range(4):
                # Target square coords
                tx = tax + WALL_BLOCK_OFFSETS[j, 0]
                ty = tay + WALL_BLOCK_OFFSETS[j, 1]
                tz = taz + WALL_BLOCK_OFFSETS[j, 2]
                
                # Bounds check before array access
                if not (0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE):
                    is_blocked = True
                    break
                
                # Direct occupancy lookup
                occ_color = occ[tx, ty, tz]
                
                if occ_color != 0:
                    # Square is occupied.
                    # It blocks IF:
                    # 1. It is friendly (occ_color == my_color)
                    # 2. AND it is NOT part of the current wall (self-overlap)
                    
                    if occ_color == my_color:
                        # Check self-overlap
                        is_self = False
                        for k in range(4):
                            if (current_squares[k, 0] == tx and 
                                current_squares[k, 1] == ty and 
                                current_squares[k, 2] == tz):
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

def generate_wall_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    anchors = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if anchors.ndim == 1:
        anchors = anchors.reshape(1, 3)
        
    if anchors.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Filter out non-anchor wall parts
    # A wall piece is an anchor ONLY if it has no wall to its left (x-1) and no wall above (y-1)
    # This assumes walls are 2x2 and cannot be placed immediately adjacent to each other
    # such that a new anchor is adjacent to an existing wall's parts.
    
    # Check left neighbors (x-1)
    left_coords = anchors.copy()
    left_coords[:, 0] -= 1
    
    # Check up neighbors (y-1)
    up_coords = anchors.copy()
    up_coords[:, 1] -= 1
    
    # Batch get types (handles out-of-bounds by returning 0/EMPTY)
    # We use batch_get_attributes because it includes bounds checking (returns 0 for out-of-bounds)
    # batch_get_types_only is unsafe for negative indices (wraps around)
    _, left_types = cache_manager.occupancy_cache.batch_get_attributes(left_coords)
    _, up_types = cache_manager.occupancy_cache.batch_get_attributes(up_coords)
    
    # Identify true anchors
    is_anchor = (left_types != PieceType.WALL) & (up_types != PieceType.WALL)
    
    valid_anchors = anchors[is_anchor]
    
    if valid_anchors.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    # Run kernel with direct occupancy access
    # We pass the raw 3D occupancy array
    return _generate_wall_moves_batch_kernel(
        valid_anchors, 
        cache_manager.occupancy_cache._occ, 
        color
    )

@register(PieceType.WALL)
def wall_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_wall_moves(state.cache_manager, state.color, pos)

__all__ = ["WALL_MOVEMENT_VECTORS", "generate_wall_moves", "can_capture_wall_vectorized"]
