"""Knight movement generator - 3D leaper with 24 movement vectors."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import COORD_DTYPE, Color, PieceType
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

# Piece-specific movement vectors - 24 knight movement patterns (2,1,0)
KNIGHT_MOVEMENT_VECTORS = np.array([
    [-2, -1, 0],
    [-2, 0, -1],
    [-2, 0, 1],
    [-2, 1, 0],
    [-1, -2, 0],
    [-1, 0, -2],
    [-1, 0, 2],
    [-1, 2, 0],
    [0, -2, -1],
    [0, -2, 1],
    [0, -1, -2],
    [0, -1, 2],
    [0, 1, -2],
    [0, 1, 2],
    [0, 2, -1],
    [0, 2, 1],
    [1, -2, 0],
    [1, 0, -2],
    [1, 0, 2],
    [1, 2, 0],
    [2, -1, 0],
    [2, 0, -1],
    [2, 0, 1],
    [2, 1, 0],
], dtype=COORD_DTYPE)

# Buffed knight movement vectors (2,1,1) - 48 directions
BUFFED_KNIGHT_MOVEMENT_VECTORS = np.array([
    [-2, -1, 0],
    [-2, 0, -1],
    [-2, 0, 1],
    [-2, 1, 0],
    [-1, -2, 0],
    [-1, 0, -2],
    [-1, 0, 2],
    [-1, 2, 0],
    [0, -2, -1],
    [0, -2, 1],
    [0, -1, -2],
    [0, -1, 2],
    [0, 1, -2],
    [0, 1, 2],
    [0, 2, -1],
    [0, 2, 1],
    [1, -2, 0],
    [1, 0, -2],
    [1, 0, 2],
    [1, 2, 0],
    [2, -1, 0],
    [2, 0, -1],
    [2, 0, 1],
    [2, 1, 0],
], dtype=COORD_DTYPE)

__all__ = ['KNIGHT_MOVEMENT_VECTORS', 'BUFFED_KNIGHT_MOVEMENT_VECTORS']

