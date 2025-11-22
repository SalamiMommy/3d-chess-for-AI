# Eliminate Move class entirely - use structured arrays
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import Optional
from game3d.common.shared_types import COORD_DTYPE, MOVE_DTYPE, MOVE_FLAGS

@dataclass
class Move:
    """
    Temporary Move class to satisfy existing imports.
    Wraps numpy coordinates.
    """
    from_coord: np.ndarray
    to_coord: np.ndarray
    promotion: Optional[int] = None
    
    def __post_init__(self):
        # Ensure numpy arrays
        if not isinstance(self.from_coord, np.ndarray):
            self.from_coord = np.array(self.from_coord, dtype=COORD_DTYPE)
        if not isinstance(self.to_coord, np.ndarray):
            self.to_coord = np.array(self.to_coord, dtype=COORD_DTYPE)

# Structured move dtype - no Python objects
MOVE_STRUCT_DTYPE = np.dtype([
    ('from_x', COORD_DTYPE), ('from_y', COORD_DTYPE), ('from_z', COORD_DTYPE),
    ('to_x', COORD_DTYPE), ('to_y', COORD_DTYPE), ('to_z', COORD_DTYPE),
    ('flags', np.uint8),
    ('piece_type', np.uint8),
    ('captured_type', np.uint8)
])

@njit(cache=True, parallel=True)
def create_move_structures(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    piece_types: np.ndarray,
    capture_flags: np.ndarray
) -> np.ndarray:
    """Vectorized move structure creation."""
    n = from_coords.shape[0]
    moves = np.zeros(n, dtype=MOVE_STRUCT_DTYPE)

    for i in prange(n):
        moves[i]['from_x'] = from_coords[i, 0]
        moves[i]['from_y'] = from_coords[i, 1]
        moves[i]['from_z'] = from_coords[i, 2]
        moves[i]['to_x'] = to_coords[i, 0]
        moves[i]['to_y'] = to_coords[i, 1]
        moves[i]['to_z'] = to_coords[i, 2]
        moves[i]['piece_type'] = piece_types[i]
        moves[i]['flags'] = capture_flags[i]

    return moves.view(np.ndarray).reshape(-1, MOVE_STRUCT_DTYPE.itemsize // np.dtype(np.uint8).itemsize)

