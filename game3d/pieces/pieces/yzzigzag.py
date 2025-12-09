# yzzigzag.py - FULLY NUMPY-NATIVE
"""
YZ-Zig-Zag — 9-step zig-zag rays in YZ-plane.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING, Union
import numpy as np

from game3d.common.shared_types import Color, PieceType, COORD_DTYPE

if TYPE_CHECKING: pass

def _build_yz_zigzag_vectors() -> np.ndarray:
    """Generate YZ-plane zigzag vectors using vectorized numpy operations."""
    # Generate for both primary/secondary direction combinations
    vecs = []
    for pri, sec in ((1, -1), (-1, 1)):
        # Primary and secondary steps
        primary_steps = np.tile([0, pri, 0], 3)  # Y-axis steps
        secondary_steps = np.tile([0, 0, sec], 3)  # Z-axis steps

        # Interleave primary and secondary steps
        all_steps = np.zeros(18, dtype=COORD_DTYPE)
        all_steps[0::2] = primary_steps  # Even indices (0, 2, 4...)
        all_steps[1::2] = secondary_steps  # Odd indices (1, 3, 5...)

        # Reshape to (9, 3) and accumulate
        step_array = all_steps.reshape(-1, 3)
        cumulative = np.cumsum(step_array, axis=0)
        vecs.extend(cumulative)

    return np.array(vecs, dtype=COORD_DTYPE)

YZ_ZIGZAG_DIRECTIONS = _build_yz_zigzag_vectors()

__all__ = ['YZ_ZIGZAG_DIRECTIONS']

