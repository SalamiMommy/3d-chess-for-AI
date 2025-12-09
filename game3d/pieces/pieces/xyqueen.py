# xyqueen.py - FULLY NUMPY-NATIVE
"""
XY-Queen: 8 slider rays in XY-plane + full 3-D king hop (26 directions, 1 step).
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING, Union
import numpy as np

from game3d.common.shared_types import Color, PieceType, COORD_DTYPE

if TYPE_CHECKING: pass

# 8 directions confined to the XY-plane (Z fixed)
_XY_SLIDER_DIRS = np.array([
    [-1, -1, 0],
    [-1, 0, 0],
    [-1, 1, 0],
    [0, -1, 0],
    [0, 1, 0],
    [1, -1, 0],
    [1, 0, 0],
    [1, 1, 0],
], dtype=COORD_DTYPE)

# 26 one-step king directions (3-D) - converted to numpy-native using meshgrid
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
# FIXED: Use np.any to keep rows where AT LEAST ONE coord is non-zero
origin_mask = np.any(all_coords != 0, axis=1)
_KING_3D_DIRS = all_coords[origin_mask].astype(COORD_DTYPE)

__all__ = ['_XY_SLIDER_DIRS', '_KING_3D_DIRS']

