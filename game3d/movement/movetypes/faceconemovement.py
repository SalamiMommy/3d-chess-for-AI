"""3D Face Cone Slider — conical sliding rays with symmetry optimisation."""

import numpy as np
from typing import List, Set, Tuple
from math import gcd
from game3d.pieces.enums import PieceType, Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.slidermovement import get_slider_generator
from game3d.cache.manager import OptimizedCacheManager

# --------------------------------------------------------------------------- #
#  Geometry owned by this module                                             #
# --------------------------------------------------------------------------- #
def _precompute_cone_directions() -> List[Tuple[int, int, int]]:
    directions: Set[Tuple[int, int, int]] = set()
    MAX_D = 8
    cones = [
        lambda dx, dy, dz: dx > 0 and abs(dy) <= dx and abs(dz) <= dx,  # +X
        lambda dx, dy, dz: dx < 0 and abs(dy) <= -dx and abs(dz) <= -dx, # -X
        lambda dx, dy, dz: dy > 0 and abs(dx) <= dy and abs(dz) <= dy,  # +Y
        lambda dx, dy, dz: dy < 0 and abs(dx) <= -dy and abs(dz) <= -dy, # -Y
        lambda dx, dy, dz: dz > 0 and abs(dx) <= dz and abs(dy) <= dz,  # +Z
        lambda dx, dy, dz: dz < 0 and abs(dx) <= -dz and abs(dy) <= -dz, # -Z
    ]
    for cone in cones:
        for dx in range(-MAX_D, MAX_D + 1):
            for dy in range(-MAX_D, MAX_D + 1):
                for dz in range(-MAX_D, MAX_D + 1):
                    if dx == dy == dz == 0:
                        continue
                    if not cone(dx, dy, dz):
                        continue
                    g = gcd(gcd(abs(dx), abs(dy)), abs(dz))
                    if g > 0:
                        prim = (dx // g, dy // g, dz // g)
                        directions.add(prim)
    return list(directions)

CONE_DIRECTIONS = np.array(_precompute_cone_directions())

# --------------------------------------------------------------------------- #
#  Public API — signature unchanged                                          #
# --------------------------------------------------------------------------- #
def generate_face_cone_slider_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int,
    y: int,
    z: int
) -> List[Move]:
    engine = get_slider_generator(cache)
    return engine.generate(
        color=color,
        ptype=PieceType.CONESLIDER,   # <-- NEW
        pos=(x, y, z),
        directions=CONE_DIRECTIONS,
        max_steps=8,
       
        
        
        
    )
