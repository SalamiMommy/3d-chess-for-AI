# yzqueen.py - FULLY NUMPY-NATIVE
"""
YZ-Queen: 8 slider rays in YZ-plane + full 3-D king hop (26 directions, 1 step).
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

# 8 directions confined to the YZ-plane (X fixed)
_YZ_SLIDER_DIRS = np.array([
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
    [0, 1, 1], [0, 1, -1],
    [0, -1, 1], [0, -1, -1]
], dtype=COORD_DTYPE)

# 26 one-step king directions (3-D) - converted to numpy-native using meshgrid
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
# Remove the (0, 0, 0) origin
# FIXED: Use np.any to keep rows where AT LEAST ONE coord is non-zero
origin_mask = np.any(all_coords != 0, axis=1)
_KING_3D_DIRS = all_coords[origin_mask].astype(COORD_DTYPE)

def generate_yz_queen_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    max_steps: Union[int, np.ndarray] = 8,
    ignore_occupancy: bool = False
) -> np.ndarray:
    """Slider rays (YZ-plane, 8 dirs, 8 steps) + king hop (26 dirs, 1 step)."""
    pos_arr = pos.astype(COORD_DTYPE)

    # Validate position
    if pos_arr.ndim == 1:
        # Lazy import to avoid circular dependency
        from game3d.common.coord_utils import in_bounds_vectorized
        if not in_bounds_vectorized(pos_arr.reshape(1, 3))[0]:
            return np.empty((0, 6), dtype=COORD_DTYPE)

    # Get slider engine instance
    slider_engine = get_slider_movement_generator()
    move_arrays = []

    # Generate slider moves in YZ-plane
    slider_moves = slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=_YZ_SLIDER_DIRS,
        max_distance=max_steps,
        ignore_occupancy=ignore_occupancy
    )
    if slider_moves.size > 0:
        move_arrays.append(slider_moves)

    # Generate king hop moves (3D)
    king_moves = slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=_KING_3D_DIRS,
        max_distance=1,
        ignore_occupancy=ignore_occupancy
    )
    if king_moves.size > 0:
        move_arrays.append(king_moves)

    if not move_arrays:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    return np.concatenate(move_arrays)

@register(PieceType.YZQUEEN)
def yz_queen_move_dispatcher(state: 'GameState', pos: np.ndarray, ignore_occupancy: bool = False) -> np.ndarray:
    """Registered dispatcher for YZ-Queen moves."""
    return generate_yz_queen_moves(state.cache_manager, state.color, pos, 8, ignore_occupancy)

__all__ = ['_YZ_SLIDER_DIRS', '_KING_3D_DIRS', 'generate_yz_queen_moves']
