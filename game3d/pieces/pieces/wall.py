# game3d/movement/movetypes/wall.py - OPTIMIZED NUMBA VERSION
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
def _generate_wall_moves_kernel(
    anchor: np.ndarray, 
    colors: np.ndarray, 
    types: np.ndarray,
    my_color: int
) -> np.ndarray:
    """
    Fused kernel for wall move generation.
    Input:
        anchor: (3,)
        colors: (6, 4) - colors at target squares for 6 directions
        types: (6, 4) - types at target squares
        my_color: int
    Returns:
        (N, 6) moves
    """
    moves = np.empty((6, 6), dtype=COORD_DTYPE)
    count = 0
    
    # Current wall squares for overlap check
    current_squares = np.empty((4, 3), dtype=COORD_DTYPE)
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
            
            # Occupancy at target square
            occ_color = colors[i, j]
            
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
    anchor = pos.astype(COORD_DTYPE)

    # 1. Enforce Anchor Logic
    if not is_wall_anchor(anchor, cache_manager):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    if not _block_in_bounds_numba(anchor):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # 2. Prepare data for kernel
    # We need to fetch occupancy for all 6 potential moves * 4 squares
    # Total 24 squares to check.
    
    # Generate all target squares
    # (6, 4, 3)
    target_anchors = anchor + WALL_MOVEMENT_VECTORS
    all_targets = target_anchors[:, np.newaxis, :] + WALL_BLOCK_OFFSETS[np.newaxis, :, :]
    
    # Flatten for batch lookup
    flat_targets = all_targets.reshape(-1, 3)
    
    # Batch lookup (unsafe is fine as we handle bounds in kernel? 
    # No, we need valid coords for lookup.
    # The kernel checks bounds for the *anchor*, but we need to check bounds for *squares* before lookup?
    # Actually, if anchor is in bounds (0..SIZE-2), then anchor+offset (0..1) is in bounds (0..SIZE-1).
    # So if target anchor is valid, all its squares are valid.
    
    # But fetching for INVALID target anchors might be OOB.
    # So we should filter valid anchors first, OR use safe lookup.
    # Safe lookup is easier.
    
    colors, types = cache_manager.occupancy_cache.batch_get_attributes(flat_targets)
    
    # Reshape
    colors = colors.reshape(6, 4)
    types = types.reshape(6, 4)
    
    # 3. Run kernel
    return _generate_wall_moves_kernel(anchor, colors, types, color)

@register(PieceType.WALL)
def wall_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_wall_moves(state.cache_manager, state.color, pos)

__all__ = ["WALL_MOVEMENT_VECTORS", "generate_wall_moves", "can_capture_wall_vectorized"]
