"""King movement generator - 26-direction king/priest piece."""
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import *
from game3d.common.shared_types import Color, PieceType
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING: pass

# Piece-specific movement vectors - 26 directions (3x3x3 - 1)
# Converted to numpy-native using meshgrid for better performance
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
origin_mask = np.any(all_coords != 0, axis=1)
KING_MOVEMENT_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

# Buffed King: 5x5x5 cube (Chebyshev distance 2)
dx_vals_b, dy_vals_b, dz_vals_b = np.meshgrid([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], indexing='ij')
all_coords_b = np.stack([dx_vals_b.ravel(), dy_vals_b.ravel(), dz_vals_b.ravel()], axis=1)
origin_mask_b = np.any(all_coords_b != 0, axis=1)
BUFFED_KING_MOVEMENT_VECTORS = all_coords_b[origin_mask_b].astype(COORD_DTYPE)

__all__ = ['KING_MOVEMENT_VECTORS', 'BUFFED_KING_MOVEMENT_VECTORS']

