"""
Echo piece implementation - 1-sphere surface projection with ±2 axis offset.

The Echo piece moves along a 1-sphere surface by projecting from 6 cardinal anchor points
(offset by 2 spaces) and adding 26 radius-1 bubble offsets.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import *

if TYPE_CHECKING: pass

# Echo piece-specific movement vectors (numpy arrays)
# 6 cardinal anchors at offset 2 (unbuffed)
_ANCHORS = np.array([
    [-2, -1, -1],
    [-2, -1, 0],
    [-2, -1, 1],
    [-2, 0, -1],
    [-2, 0, 0],
    [-2, 0, 1],
    [-2, 1, -1],
    [-2, 1, 0],
    [-2, 1, 1],
    [-1, -2, -1],
    [-1, -2, 0],
    [-1, -2, 1],
    [-1, -1, -2],
    [-1, -1, 2],
    [-1, 0, -2],
    [-1, 0, 2],
    [-1, 1, -2],
    [-1, 1, 2],
    [-1, 2, -1],
    [-1, 2, 0],
    [-1, 2, 1],
    [0, -2, -1],
    [0, -2, 0],
    [0, -2, 1],
    [0, -1, -2],
    [0, -1, 2],
    [0, 0, -2],
    [0, 0, 2],
    [0, 1, -2],
    [0, 1, 2],
    [0, 2, -1],
    [0, 2, 0],
    [0, 2, 1],
    [1, -2, -1],
    [1, -2, 0],
    [1, -2, 1],
    [1, -1, -2],
    [1, -1, 2],
    [1, 0, -2],
    [1, 0, 2],
    [1, 1, -2],
    [1, 1, 2],
    [1, 2, -1],
    [1, 2, 0],
    [1, 2, 1],
    [2, -1, -1],
    [2, -1, 0],
    [2, -1, 1],
    [2, 0, -1],
    [2, 0, 0],
    [2, 0, 1],
    [2, 1, -1],
    [2, 1, 0],
    [2, 1, 1],
], dtype=COORD_DTYPE)

# 6 cardinal anchors at offset 3 (buffed - 1 space further)
_BUFFED_ANCHORS = np.array([
    [-2, -1, -1],
    [-2, -1, 0],
    [-2, -1, 1],
    [-2, 0, -1],
    [-2, 0, 0],
    [-2, 0, 1],
    [-2, 1, -1],
    [-2, 1, 0],
    [-2, 1, 1],
    [-1, -2, -1],
    [-1, -2, 0],
    [-1, -2, 1],
    [-1, -1, -2],
    [-1, -1, 2],
    [-1, 0, -2],
    [-1, 0, 2],
    [-1, 1, -2],
    [-1, 1, 2],
    [-1, 2, -1],
    [-1, 2, 0],
    [-1, 2, 1],
    [0, -2, -1],
    [0, -2, 0],
    [0, -2, 1],
    [0, -1, -2],
    [0, -1, 2],
    [0, 0, -2],
    [0, 0, 2],
    [0, 1, -2],
    [0, 1, 2],
    [0, 2, -1],
    [0, 2, 0],
    [0, 2, 1],
    [1, -2, -1],
    [1, -2, 0],
    [1, -2, 1],
    [1, -1, -2],
    [1, -1, 2],
    [1, 0, -2],
    [1, 0, 2],
    [1, 1, -2],
    [1, 1, 2],
    [1, 2, -1],
    [1, 2, 0],
    [1, 2, 1],
    [2, -1, -1],
    [2, -1, 0],
    [2, -1, 1],
    [2, 0, -1],
    [2, 0, 0],
    [2, 0, 1],
    [2, 1, -1],
    [2, 1, 0],
    [2, 1, 1],
], dtype=COORD_DTYPE)

# 26 radius-1 bubble offsets
_BUBBLE = RADIUS_1_OFFSETS.copy()

# 156 raw jump vectors (anchors + bubbles) - unbuffed
_ECHO_DIRECTIONS = (_ANCHORS[:, None, :] + _BUBBLE[None, :, :]).reshape(-1, 3)

# 156 raw jump vectors (buffed anchors + bubbles) - buffed
_BUFFED_ECHO_DIRECTIONS = (_BUFFED_ANCHORS[:, None, :] + _BUBBLE[None, :, :]).reshape(-1, 3)

# Public constants for export
ECHO_MOVEMENT_VECTORS = _ECHO_DIRECTIONS
BUFFED_ECHO_MOVEMENT_VECTORS = _BUFFED_ECHO_DIRECTIONS

__all__ = ['ECHO_MOVEMENT_VECTORS', 'BUFFED_ECHO_MOVEMENT_VECTORS']

