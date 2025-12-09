"""Bishop movement generator - diagonal slider."""
import numpy as np
from typing import List, TYPE_CHECKING, Union

from game3d.common.coord_utils import CoordinateUtils
from game3d.common.shared_types import COORD_DTYPE, PieceType, MAX_STEPS_SLIDER

if TYPE_CHECKING: pass

# Piece-specific movement vectors - 12 diagonal directions for 3D
BISHOP_MOVEMENT_VECTORS = np.array([
    [-1, -1, 0],
    [-1, 0, -1],
    [-1, 0, 1],
    [-1, 1, 0],
    [0, -1, -1],
    [0, -1, 1],
    [0, 1, -1],
    [0, 1, 1],
    [1, -1, 0],
    [1, 0, -1],
    [1, 0, 1],
    [1, 1, 0],
], dtype=COORD_DTYPE)

__all__ = ['BISHOP_MOVEMENT_VECTORS']

