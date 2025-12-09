"""Big Knights - Extended knight pieces with (3,1) and (3,2) leap patterns."""
from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import *
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

# Piece-specific movement vectors - Knight31 (3,1,0) leap pattern
KNIGHT31_MOVEMENT_VECTORS = np.array([
    [-3, -1, 0],
    [-3, 0, -1],
    [-3, 0, 1],
    [-3, 1, 0],
    [-1, -3, 0],
    [-1, 0, -3],
    [-1, 0, 3],
    [-1, 3, 0],
    [0, -3, -1],
    [0, -3, 1],
    [0, -1, -3],
    [0, -1, 3],
    [0, 1, -3],
    [0, 1, 3],
    [0, 3, -1],
    [0, 3, 1],
    [1, -3, 0],
    [1, 0, -3],
    [1, 0, 3],
    [1, 3, 0],
    [3, -1, 0],
    [3, 0, -1],
    [3, 0, 1],
    [3, 1, 0],
], dtype=COORD_DTYPE)

# Buffed Knight31 (3,1,1) leap pattern
BUFFED_KNIGHT31_MOVEMENT_VECTORS = np.array([
    [-3, -1, 0],
    [-3, 0, -1],
    [-3, 0, 1],
    [-3, 1, 0],
    [-1, -3, 0],
    [-1, 0, -3],
    [-1, 0, 3],
    [-1, 3, 0],
    [0, -3, -1],
    [0, -3, 1],
    [0, -1, -3],
    [0, -1, 3],
    [0, 1, -3],
    [0, 1, 3],
    [0, 3, -1],
    [0, 3, 1],
    [1, -3, 0],
    [1, 0, -3],
    [1, 0, 3],
    [1, 3, 0],
    [3, -1, 0],
    [3, 0, -1],
    [3, 0, 1],
    [3, 1, 0],
], dtype=COORD_DTYPE)

# Piece-specific movement vectors - Knight32 (3,2,0) leap pattern
KNIGHT32_MOVEMENT_VECTORS = np.array([
    [-3, -2, 0],
    [-3, 0, -2],
    [-3, 0, 2],
    [-3, 2, 0],
    [-2, -3, 0],
    [-2, 0, -3],
    [-2, 0, 3],
    [-2, 3, 0],
    [0, -3, -2],
    [0, -3, 2],
    [0, -2, -3],
    [0, -2, 3],
    [0, 2, -3],
    [0, 2, 3],
    [0, 3, -2],
    [0, 3, 2],
    [2, -3, 0],
    [2, 0, -3],
    [2, 0, 3],
    [2, 3, 0],
    [3, -2, 0],
    [3, 0, -2],
    [3, 0, 2],
    [3, 2, 0],
], dtype=COORD_DTYPE)

# Buffed Knight32 (3,2,1) leap pattern
BUFFED_KNIGHT32_MOVEMENT_VECTORS = np.array([
    [-3, -2, 0],
    [-3, 0, -2],
    [-3, 0, 2],
    [-3, 2, 0],
    [-2, -3, 0],
    [-2, 0, -3],
    [-2, 0, 3],
    [-2, 3, 0],
    [0, -3, -2],
    [0, -3, 2],
    [0, -2, -3],
    [0, -2, 3],
    [0, 2, -3],
    [0, 2, 3],
    [0, 3, -2],
    [0, 3, 2],
    [2, -3, 0],
    [2, 0, -3],
    [2, 0, 3],
    [2, 3, 0],
    [3, -2, 0],
    [3, 0, -2],
    [3, 0, 2],
    [3, 2, 0],
], dtype=COORD_DTYPE)

__all__ = []

