
"""
Stateless Attack and Check Detection.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple

from game3d.core.buffer import GameBuffer
from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE,
    SIZE, Color, PieceType
)
from game3d.core.moving import apply_move, apply_move_inplace, undo_move_inplace

# Import kernels
from game3d.attacks.fast_attack import _fast_attack_kernel

# DUPLICATED HELPER
@njit(cache=True)
def _compute_buffs_local(occupied_types: np.ndarray, occupied_coords: np.ndarray, count: int) -> np.ndarray:
    is_buffed = np.zeros((SIZE, SIZE, SIZE), dtype=np.bool_)
    REFLECTOR_TYPE = 35 
    # Hardcoded King Vectors for buff spread
    vectors = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=COORD_DTYPE)

    for i in range(count):
        if occupied_types[i] == REFLECTOR_TYPE:
            rx, ry, rz = occupied_coords[i]
            for d in range(vectors.shape[0]):
                dx, dy, dz = vectors[d]
                bx, by, bz = rx + dx, ry + dy, rz + dz
                if 0 <= bx < SIZE and 0 <= by < SIZE and 0 <= bz < SIZE:
                    is_buffed[bx, by, bz] = True
    return is_buffed

@njit(cache=True)
def find_king(buffer: GameBuffer, color: int) -> np.ndarray:
    """Find king position. Uses meta index or scan."""
    # Use meta index
    idx = -1
    if color == 1:
        idx = buffer.meta[4]
    else:
        idx = buffer.meta[5]
        
    # Verify index validity (robustness)
    if 0 <= idx < buffer.occupied_count:
        if buffer.occupied_types[idx] == 6 and buffer.occupied_colors[idx] == color:
            return buffer.occupied_coords[idx]

    # Fallback Scan
    for i in range(buffer.occupied_count):
        if buffer.occupied_types[i] == 6 and buffer.occupied_colors[i] == color:
            # Update meta? No, buffer is immutable-ish here (passed by val)
            return buffer.occupied_coords[i]
            
    # Should not happen in legal game
    return np.array([-1, -1, -1], dtype=COORD_DTYPE)

@njit(cache=True)
def is_square_attacked(buffer: GameBuffer, square: np.ndarray, attacker_color: int, cached_moves: np.ndarray = None) -> bool:
    """Check if square is attacked by attacker_color."""
    
    # 1. OPTIMIZATION: Use cached moves if available
    # Check if cached_moves is not None and not empty
    # In Numba, optional arrays are tricky. We assume cached_moves is always passed, 
    # potentially as empty array if no cache.
    if cached_moves is not None and cached_moves.size > 0:
        # Check if any move targets the square (Simple linear scan - Vectorizable?)
        # cached_moves is (N, 6)
        # Target match: moves[:, 3:6] == square
        
        # Numba efficient loop
        sx, sy, sz = square[0], square[1], square[2]
        n_cached = cached_moves.shape[0]
        for i in range(n_cached):
            if cached_moves[i, 3] == sx and cached_moves[i, 4] == sy and cached_moves[i, 5] == sz:
                return True
        
        # NOTE: If we are using cache, we assume cache is COMPLETE for the current board state.
        # If no move in cache targets square, it's not attacked.
        return False
        
    # 2. Fallback: Recalculate using kernel (Expensive)
    
    # We need attacker pieces
    # Filter arrays
    occ_count = buffer.occupied_count
    
    # We need arrays of attacker pieces for the kernel
    # Allocate max possible (count)
    atk_coords = np.empty((occ_count, 3), dtype=COORD_DTYPE)
    atk_types = np.empty(occ_count, dtype=np.int8) 
    
    count = 0
    for i in range(occ_count):
        if buffer.occupied_colors[i] == attacker_color:
            atk_coords[count] = buffer.occupied_coords[i]
            atk_types[count] = buffer.occupied_types[i]
            count += 1
            
    if count == 0:
        return False
        
    # Call kernel
    skipped = np.zeros(count + 1, dtype=np.int32)
    
    result = _fast_attack_kernel(
        square,
        atk_coords[:count],
        atk_types[:count],
        buffer.board_color, # occ grid
        attacker_color,
        skipped
    )
    
    return result == 1 

@njit(cache=True)
def is_check(buffer: GameBuffer, color: int, cached_opponent_moves: np.ndarray = None) -> bool:
    """Is the king of 'color' in check?"""
    # ✅ CHECK FOR PRIESTS
    if color == 1:
        if buffer.meta[6] > 0:
            return False
    else:
        if buffer.meta[7] > 0:
            return False

    king_pos = find_king(buffer, color)
    if king_pos[0] == -1:
        return True # No king = lost?
        
    attacker_color = 1 if color == 2 else 2
    return is_square_attacked(buffer, king_pos, attacker_color, cached_opponent_moves)

@njit(cache=True)
def filter_legal_moves(buffer: GameBuffer, moves: np.ndarray) -> np.ndarray:
    """
    Filter moves that leave king in check.
    Uses in-place apply/undo for zero-allocation filtering.
    """
    n_moves = moves.shape[0]
    if n_moves == 0:
        return moves
        
    legal_mask = np.zeros(n_moves, dtype=BOOL_DTYPE)
    active_color = buffer.meta[0]
    
    # Empty cache for loop (board is modified, so static cache is invalid)
    empty_cache = np.empty((0, 6), dtype=COORD_DTYPE)
    
    for i in range(n_moves):
        # Apply move in-place
        undo_info = apply_move_inplace(buffer, moves[i])
        
        # Check if WE are in check
        # NOTE: caching is disabled here because 'is_check' runs on modified board
        if not is_check(buffer, active_color, empty_cache):
            legal_mask[i] = True
        
        # Undo move in-place
        undo_move_inplace(buffer, undo_info)
            
    # Filter
    count = np.sum(legal_mask)
    legal_moves = np.empty((count, 6), dtype=COORD_DTYPE)
    
    ptr = 0
    for i in range(n_moves):
        if legal_mask[i]:
            legal_moves[ptr] = moves[i]
            ptr += 1
            
    return legal_moves

