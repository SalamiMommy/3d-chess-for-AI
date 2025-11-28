# spiral.py - FULLY NUMPY-NATIVE
"""
Spiral-Slider — 6 counter-clockwise spiral rays.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.common.registry import register
from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Maximum movement distance for spiral piece (matches board size)
MAX_SPIRAL_DISTANCE = 8

# Piece-specific movement vectors as numpy arrays - 6 spiral directions
SPIRAL_MOVEMENT_VECTORS = np.array([
    # Main direction: +X, 8 spiral steps
    [ 1,  0,  0], [ 0,  1,  0], [-1,  1,  0], [-1,  0,  0],
    [-1, -1,  0], [ 0, -1,  0], [ 1, -1,  0], [ 1,  0,  0],
    # Main direction: -X, 8 spiral steps
    [-1,  0,  0], [ 0, -1,  0], [-1, -1,  0], [-1,  0,  0],
    [-1,  1,  0], [ 0,  1,  0], [ 1,  1,  0], [ 1,  0,  0],
    # Main direction: +Y, 8 spiral steps
    [ 0,  1,  0], [-1,  0,  0], [-1,  0,  1], [ 0,  0,  1],
    [ 1,  0,  1], [ 1,  0,  0], [ 1,  0, -1], [ 0,  0, -1],
    # Main direction: -Y, 8 spiral steps
    [ 0, -1,  0], [ 1,  0,  0], [ 1,  0,  1], [ 0,  0,  1],
    [-1,  0,  1], [-1,  0,  0], [-1,  0, -1], [ 0,  0, -1],
    # Main direction: +Z, 8 spiral steps
    [ 0,  0,  1], [ 1,  0,  0], [ 1,  1,  0], [ 0,  1,  0],
    [-1,  1,  0], [-1,  0,  0], [-1, -1,  0], [ 0, -1,  0],
    # Main direction: -Z, 8 spiral steps
    [ 0,  0, -1], [-1,  0,  0], [-1,  1,  0], [ 0,  1,  0],
    [ 1,  1,  0], [ 1,  0,  0], [ 1, -1,  0], [ 0, -1,  0],
], dtype=COORD_DTYPE)

def generate_spiral_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray,
    ignore_occupancy: bool = False
) -> np.ndarray:
    """Generate spiral moves from numpy-native position array."""
    pos_arr = pos.astype(COORD_DTYPE)
    slider_engine = get_slider_movement_generator()
    return slider_engine.generate_slider_moves_array(
        cache_manager=cache_manager,
        color=color,
        pos=pos_arr,
        directions=SPIRAL_MOVEMENT_VECTORS,
        max_distance=MAX_SPIRAL_DISTANCE,
        ignore_occupancy=ignore_occupancy
    )

@register(PieceType.SPIRAL)
def spiral_move_dispatcher(state: 'GameState', pos: np.ndarray, ignore_occupancy: bool = False) -> np.ndarray:
    """Registered dispatcher for Spiral moves."""
    return generate_spiral_moves(state.cache_manager, state.color, pos, ignore_occupancy)

__all__ = ['SPIRAL_MOVEMENT_VECTORS', 'MAX_SPIRAL_DISTANCE', 'generate_spiral_moves', 'spiral_move_dispatcher']
