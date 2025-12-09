"""Panel movement generator - teleport on same x/y/z plane + king moves."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_MINUS_1
from game3d.common.shared_types import Color, PieceType
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

# Piece-specific movement vectors - 3x3 panels at distance 2 (unbuffed) and 3 (buffed)

def _create_panel_vectors(distance):
    vectors = []
    r = [-1, 0, 1]
    
    # X faces
    for x in [-distance, distance]:
        for y in r:
            for z in r:
                vectors.append([x, y, z])
                
    # Y faces
    for y in [-distance, distance]:
        for x in r:
            for z in r:
                vectors.append([x, y, z])
                
    # Z faces
    for z in [-distance, distance]:
        for x in r:
            for y in r:
                vectors.append([x, y, z])
                
    return np.array(vectors, dtype=COORD_DTYPE)

PANEL_MOVEMENT_VECTORS = _create_panel_vectors(2)
BUFFED_PANEL_MOVEMENT_VECTORS = _create_panel_vectors(3)

__all__ = ['PANEL_MOVEMENT_VECTORS']

