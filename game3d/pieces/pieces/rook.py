"""Rook movement generator - orthogonal slider with 6 directions."""
import numpy as np
from typing import List, TYPE_CHECKING, Union

from game3d.common.shared_types import COORD_DTYPE, Color, PieceType
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

# Piece-specific movement vectors - 6 orthogonal directions
ROOK_MOVEMENT_VECTORS = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
], dtype=COORD_DTYPE)

__all__ = ['ROOK_MOVEMENT_VECTORS']

