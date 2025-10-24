"""
Face-Cone-Slider — 6 conical rays (consolidated).
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
from math import gcd
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.slidermovement import generate_moves
from game3d.movement.movepiece import Move
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def _cone_dirs() -> np.ndarray:
    """Optimized cone direction generation."""
    dirs = set()
    cone_axes = [
        (1, 0, 0), (-1, 0, 0),  # +X, -X
        (0, 1, 0), (0, -1, 0),  # +Y, -Y
        (0, 0, 1), (0, 0, -1)   # +Z, -Z
    ]

    for primary_axis in cone_axes:
        px, py, pz = primary_axis
        for dy in range(-8, 9):
            for dz in range(-8, 9):
                if px != 0:  # X-cone
                    if abs(dy) <= abs(px * 8) and abs(dz) <= abs(px * 8):
                        dx = px * max(1, abs(dy), abs(dz))
                        dirs.add((dx, dy, dz))
                elif py != 0:  # Y-cone
                    if abs(dx) <= abs(py * 8) and abs(dz) <= abs(py * 8):
                        dy = py * max(1, abs(dx), abs(dz))
                        dirs.add((dx, dy, dz))
                else:  # Z-cone
                    if abs(dx) <= abs(pz * 8) and abs(dy) <= abs(pz * 8):
                        dz = pz * max(1, abs(dx), abs(dy))
                        dirs.add((dx, dy, dz))

    primitive_dirs = set()
    for dx, dy, dz in dirs:
        if dx == dy == dz == 0:
            continue
        g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
        primitive_dirs.add((dx // g, dy // g, dz // g))

    return np.array(list(primitive_dirs), dtype=np.int8)

CONE_DIRECTIONS = _cone_dirs()

def generate_face_cone_slider_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_moves(
        piece_type='cone_slider',
        pos=(x, y, z),
        color=color,
        max_distance=16,
        directions=CONE_DIRECTIONS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

@register(PieceType.CONESLIDER)
def face_cone_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_face_cone_slider_moves(state.cache_manager, state.color, x, y, z)  # FIXED: Use cache_manager

__all__ = ['generate_face_cone_slider_moves']
