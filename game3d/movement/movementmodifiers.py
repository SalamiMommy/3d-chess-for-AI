"""Fully optimized movement modifiers - now matches generator.py (N,6) architecture.

This module works exclusively with coordinate arrays of shape (N, 6) where each
row is [from_x, from_y, from_z, to_x, to_y, to_z]. Flags and metadata are
handled separately or deferred to validation/Move object creation.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, Union, TYPE_CHECKING
from game3d.common.shared_types import COORD_DTYPE, BOOL_DTYPE, INDEX_DTYPE, SIZE, MOVE_FLAGS
from game3d.common.coord_utils import in_bounds_vectorized

# =============================================================================
# CORE FILTERING - WORKS WITH (N, 6) ARRAYS
# =============================================================================
@njit(cache=True, fastmath=True, parallel=True)
def filter_valid_moves(move_coords: np.ndarray) -> np.ndarray:
    """Filter (N, 6) move arrays to only valid in-bounds coordinates."""
    if move_coords.shape[0] == 0:
        return move_coords

    n = move_coords.shape[0]
    valid_mask = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        # Check from coordinates
        from_valid = (0 <= move_coords[i, 0] < SIZE and
                      0 <= move_coords[i, 1] < SIZE and
                      0 <= move_coords[i, 2] < SIZE)

        # Check to coordinates
        to_valid = (0 <= move_coords[i, 3] < SIZE and
                    0 <= move_coords[i, 4] < SIZE and
                    0 <= move_coords[i, 5] < SIZE)

        valid_mask[i] = from_valid and to_valid

    return move_coords[valid_mask]

# =============================================================================
# EFFECT APPLICATION - SIMPLIFIED FOR (N, 6) ARRAYS
# =============================================================================
def apply_buff_effects_vectorized(move_coords: np.ndarray,
                                 effect_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply buff effects to moves. Placeholder for coordinate modifications."""
    # For now, return copy. Implement actual coordinate transformations here.
    return move_coords.copy() if move_coords.size > 0 else move_coords

def apply_debuff_effects_vectorized(move_coords: np.ndarray,
                                   effect_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply debuff effects to moves. Placeholder for coordinate filtering."""
    # For now, return copy. Implement actual move filtering here.
    return move_coords.copy() if move_coords.size > 0 else move_coords

def apply_capture_markers_vectorized(move_coords: np.ndarray,
                                    capture_mask: np.ndarray) -> np.ndarray:
    """Mark capture moves - returns unchanged coordinates (flags handled separately)."""
    if move_coords.size > 0 and capture_mask.shape[0] != move_coords.shape[0]:
        raise ValueError(f"Mask shape mismatch: {capture_mask.shape[0]} != {move_coords.shape[0]}")
    return move_coords.copy() if move_coords.size > 0 else move_coords

# =============================================================================
# ANALYSIS FUNCTIONS - WORK WITH SEPARATE FLAG ARRAYS
# =============================================================================
def get_move_count(move_coords: np.ndarray) -> int:
    return move_coords.shape[0]

def get_capture_count(capture_flags: np.ndarray) -> int:
    return np.sum(capture_flags) if capture_flags.size > 0 else 0

def get_buffed_count(buff_flags: np.ndarray) -> int:
    return np.sum(buff_flags) if buff_flags.size > 0 else 0

def get_debuffed_count(debuff_flags: np.ndarray) -> int:
    return np.sum(debuff_flags) if debuff_flags.size > 0 else 0

# =============================================================================
# COORDINATE EXTRACTION
# =============================================================================
def extract_from_coords(move_coords: np.ndarray) -> np.ndarray:
    return move_coords[:, :3]

def extract_to_coords(move_coords: np.ndarray) -> np.ndarray:
    return move_coords[:, 3:]

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    'filter_valid_moves',
    'apply_buff_effects_vectorized',
    'apply_debuff_effects_vectorized',
    'apply_capture_markers_vectorized',
    'get_move_count',
    'get_capture_count',
    'get_buffed_count',
    'get_debuffed_count',
    'extract_from_coords',
    'extract_to_coords',
    'extract_to_coords',
    'MOVE_FLAGS',
    'get_range_modifier'
]


