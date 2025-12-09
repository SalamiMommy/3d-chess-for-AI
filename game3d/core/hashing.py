
"""
Stateless Zobrist Hashing.
"""

import numpy as np
from numba import njit
from game3d.common.shared_types import (
    SIZE, N_PIECE_TYPES, HASH_DTYPE
)
from game3d.core.buffer import GameBuffer

# Initialize Zobrist Table (Deterministic)
# Shape: (2, 64, SIZE, SIZE, SIZE) -> Color, PieceType, X, Y, Z
# Colors: 0=White (1), 1=Black (2) - mapped
# PieceTypes: 0..63
np.random.seed(42)
ZOBRIST_TABLE = np.random.randint(
    0, 2**64, 
    (2, 64, SIZE, SIZE, SIZE), 
    dtype=np.uint64
).astype(HASH_DTYPE)

SIDE_TO_MOVE_KEY = np.random.randint(0, 2**64, dtype=np.uint64).astype(HASH_DTYPE)

@njit(cache=True)
def get_zobrist_key(color_idx: int, piece_type: int, x: int, y: int, z: int) -> int:
    """Get Zobrist key for a specific piece at specific square."""
    # Color idx: 0 for White (1), 1 for Black (2)
    return ZOBRIST_TABLE[color_idx, piece_type, x, y, z]

@njit(cache=True)
def compute_hash_from_buffer(buffer) -> int:
    """Compute Zobrist hash from scratch from a GameBuffer."""
    h = 0
    # Use sparse iteration
    for i in range(buffer.occupied_count):
        # Unpack
        pt = buffer.occupied_types[i]
        c = buffer.occupied_colors[i]
        x = buffer.occupied_coords[i, 0]
        y = buffer.occupied_coords[i, 1]
        z = buffer.occupied_coords[i, 2]
        
        # Color: 1->0, 2->1
        c_idx = c - 1
        h ^= ZOBRIST_TABLE[c_idx, pt, x, y, z]
    
    # Side to move
    # meta[0] is active color (1 or 2)
    if buffer.meta[0] == 2: # Black to move
        h ^= SIDE_TO_MOVE_KEY
        
    return h
