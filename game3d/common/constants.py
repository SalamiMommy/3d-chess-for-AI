# game3d/common/constants.py
# ------------------------------------------------------------------
# Constants and precomputed data
# ------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np

Coord = Tuple[int, int, int]

# Board geometry
SIZE = SIZE_X = SIZE_Y = SIZE_Z = 9
VOLUME = SIZE ** 3
SIZE_SQUARED = SIZE * SIZE
SIZE_MINUS_1 = SIZE - 1
# ----------------------------------------------------------
# NEW colour-aware channel budget (41 planes total)

N_PIECE_TYPES  = 40          # unchanged
N_COLOR_PLANES = 2           # unchanged
N_TOTAL_PLANES = 82
N_PLANES_PER_SIDE = N_PIECE_TYPES + N_COLOR_PLANES
# ------------------------------------------

# slices the rest of the code expects
PIECE_SLICE = slice(0, 2 * N_PIECE_TYPES)
COLOR_SLICE = slice(2 * N_PIECE_TYPES, 2 * N_PIECE_TYPES + N_COLOR_PLANES)  # 80-82
CURRENT_SLICE = slice(2 * N_PIECE_TYPES + N_COLOR_PLANES, 2 * N_PIECE_TYPES + N_COLOR_PLANES + 1)  # 82-83
EFFECT_SLICE = slice(2 * N_PIECE_TYPES + N_COLOR_PLANES + 1, 2 * N_PIECE_TYPES + N_COLOR_PLANES + 1 + 6)  # 83-89

N_CHANNELS = N_TOTAL_PLANES

# Precomputed offsets for radius=2 aura (cube minus center)
RADIUS_2_OFFSETS: List[Tuple[int, int, int]] = [
    (dx, dy, dz) for dx in range(-2, 3) for dy in range(-2, 3) for dz in range(-2, 3)
    if not (dx == dy == dz == 0)
]  # Length: 124 (5^3 - 1)

RADIUS_3_OFFSETS: List[Tuple[int, int, int]] = [
    (dx, dy, dz)
    for dx in range(-3, 4)
    for dy in range(-3, 4)
    for dz in range(-3, 4)
    if max(abs(dx), abs(dy), abs(dz)) == 3
]

_COORD_TO_IDX = {
    (x, y, z): x + 9 * y + 81 * z
    for x in range(9) for y in range(9) for z in range(9)
}
_IDX_TO_COORD = {v: k for k, v in _COORD_TO_IDX.items()}

# Batch processing constants optimized for 462 pieces
MEGA_BATCH_SIZE = 462  # Maximum pieces on board
VECTORIZED_CHUNK_SIZE = 64  # Optimal for CPU cache
MAX_CONCURRENT_BATCHES = 8

# Memory optimized arrays
OPTIMAL_DTYPE = np.int32  # Balance between memory and performance
COORD_ARRAY_SHAPE = (MEGA_BATCH_SIZE, 3)
OCCUPANCY_ARRAY_SHAPE = (SIZE_Z, SIZE_Y, SIZE_X)


