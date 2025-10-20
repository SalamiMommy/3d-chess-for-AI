"""
Face-Cone-Slider — 6 conical rays (consolidated).
Exports:
  generate_face_cone_slider_moves(cache, color, x, y, z) -> list[Move]
  (decorated) facecone_dispatcher(state, x, y, z) -> list[Move]
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING
from math import gcd
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_moves
from game3d.movement.movepiece import Move

def _cone_dirs() -> np.ndarray:
    dirs = set()
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            for dz in range(-8, 9):
                if dx == dy == dz == 0:
                    continue
                # cone predicates
                if any((
                    dx > 0 and abs(dy) <= dx and abs(dz) <= dx,
                    dx < 0 and abs(dy) <= -dx and abs(dz) <= -dx,
                    dy > 0 and abs(dx) <= dy and abs(dz) <= dy,
                    dy < 0 and abs(dx) <= -dy and abs(dz) <= -dy,
                    dz > 0 and abs(dx) <= dz and abs(dy) <= dz,
                    dz < 0 and abs(dx) <= -dz and abs(dy) <= -dz,
                )):
                    g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
                    dirs.add((dx // g, dy // g, dz // g))
    return np.array(list(dirs), dtype=np.int8)

CONE_DIRECTIONS = _cone_dirs()

def generate_face_cone_slider_moves(cache: CacheManager,
                                    color: Color,
                                    x: int, y: int, z: int) -> List[Move]:
    return generate_moves(
        piece_type='cone_slider',
        pos=(x, y, z),
        color=color.value,
        max_distance=16,
        directions=CONE_DIRECTIONS,
        occupancy=cache.occupancy._occ,
    )

@register(PieceType.CONESLIDER)
def face_cone_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_face_cone_slider_moves(state.cache, state.color, x, y, z)

__all__ = ['generate_face_cone_slider_moves']
