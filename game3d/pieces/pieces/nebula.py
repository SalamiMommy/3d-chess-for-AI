"""
Nebula piece implementation - teleport within radius-2 sphere (unbuffed) or radius-3 sphere (buffed).
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import COORD_DTYPE, COLOR_DTYPE, PieceType, RADIUS_2_OFFSETS, RADIUS_3_OFFSETS

if TYPE_CHECKING: pass

# Unbuffed: All positions within radius 2 (excluding origin)
_NEBULA_DIRECTIONS = np.array([
    [-2, 0, 0],
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -2, 0],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -2],
    [0, 0, -1],
    [0, 0, 1],
    [0, 0, 2],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [0, 2, 0],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
    [2, 0, 0],
], dtype=COORD_DTYPE)

# Buffed: All positions within radius 3 (excluding origin)
_BUFFED_NEBULA_DIRECTIONS = np.array([
    [-2, 0, 0],
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -2, 0],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -2],
    [0, 0, -1],
    [0, 0, 1],
    [0, 0, 2],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [0, 2, 0],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
    [2, 0, 0],
], dtype=COORD_DTYPE)

# Public constants for export
NEBULA_MOVEMENT_VECTORS = _NEBULA_DIRECTIONS
BUFFED_NEBULA_MOVEMENT_VECTORS = _BUFFED_NEBULA_DIRECTIONS

__all__ = ['NEBULA_MOVEMENT_VECTORS', 'BUFFED_NEBULA_MOVEMENT_VECTORS']

