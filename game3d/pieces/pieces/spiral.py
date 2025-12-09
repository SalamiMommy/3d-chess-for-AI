# spiral.py - FULLY NUMPY-NATIVE
"""
Spiral-Slider — 6 counter-clockwise spiral rays.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING, Union
import numpy as np

from game3d.common.shared_types import Color, PieceType, COORD_DTYPE

if TYPE_CHECKING: pass

# Maximum movement distance for spiral piece (matches board size)
MAX_SPIRAL_DISTANCE = 8

# Piece-specific movement vectors as numpy arrays - 6 spiral directions
SPIRAL_MOVEMENT_VECTORS = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
], dtype=COORD_DTYPE)

__all__ = ['SPIRAL_MOVEMENT_VECTORS', 'MAX_SPIRAL_DISTANCE']

