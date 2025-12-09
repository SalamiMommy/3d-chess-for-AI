"""Queen movement generator - combines rook and bishop movement."""
import numpy as np
from typing import List, TYPE_CHECKING, Union

from game3d.common.coord_utils import CoordinateUtils
from game3d.common.shared_types import COORD_DTYPE, PieceType

if TYPE_CHECKING: pass

# Piece-specific movement vectors - queen combines orthogonal + diagonal
QUEEN_MOVEMENT_VECTORS = np.array([
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
], dtype=COORD_DTYPE)

__all__ = ['QUEEN_MOVEMENT_VECTORS']

