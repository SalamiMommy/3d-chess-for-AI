# xzzigzag.py - FULLY NUMPY-NATIVE
"""
XZ-Zig-Zag — 9-step zig-zag rays in XZ-plane.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING, Union
import numpy as np

from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def _build_xz_zigzag_vectors() -> np.ndarray:
    """Generate XZ-plane zigzag vectors using vectorized numpy operations."""
    # Generate for both primary/secondary direction combinations
    vecs = []
    for pri, sec in ((1, -1), (-1, 1)):
        # Primary and secondary steps
        primary_steps = np.tile([pri, 0, 0], 3)  # X-axis steps
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

XZ_ZIGZAG_DIRECTIONS = _build_xz_zigzag_vectors()

def generate_xz_zigzag_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_steps: Union[int, np.ndarray] = 16,
    ignore_occupancy: bool = False
) -> np.ndarray:
    """Generate XZ-zigzag slider moves."""
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Validate position
    if pos_arr.ndim == 1:
        # Lazy import to avoid circular dependency
        from game3d.common.coord_utils import in_bounds_vectorized
        if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
            return np.empty((0, 6), dtype=COORD_DTYPE)

    slider_engine = get_slider_movement_generator()

    return slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=XZ_ZIGZAG_DIRECTIONS,
        max_distance=max_steps,
        ignore_occupancy=ignore_occupancy
    )

@register(PieceType.XZZIGZAG)
def xz_zigzag_move_dispatcher(state: 'GameState', pos: np.ndarray, ignore_occupancy: bool = False) -> np.ndarray:
    """Registered dispatcher for XZ-ZigZag moves."""
    return generate_xz_zigzag_moves(state.cache_manager, state.color, pos, 16, ignore_occupancy)

__all__ = ['XZ_ZIGZAG_DIRECTIONS', 'generate_xz_zigzag_moves']
