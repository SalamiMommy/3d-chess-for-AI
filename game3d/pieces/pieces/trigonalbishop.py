"""Trigonal Bishop - 8 space-diagonal movement vectors."""
from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING, Union

from game3d.common.shared_types import COORD_DTYPE, SIZE_MINUS_1, Color, PieceType
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

# Piece-specific movement vectors - 8 true 3D diagonals
TRIGONAL_BISHOP_VECTORS = np.array([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
], dtype=COORD_DTYPE)

__all__ = ['TRIGONAL_BISHOP_VECTORS']

