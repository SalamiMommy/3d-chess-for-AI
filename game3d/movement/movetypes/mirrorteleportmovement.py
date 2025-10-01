"""Mirror Teleport Move — implemented as a 1-direction jump through the
existing IntegratedJumpMovementGenerator so all caching/GPU paths stay intact.
"""

from __future__ import annotations

from typing import List
import numpy as np

from game3d.pieces.enums import Color
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.cache.manager import CacheManager

def generate_mirror_teleport_move(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """
    Teleport to (8-x, 8-y, 8-z) by treating it as a single jump direction
    and re-using the zero-redundancy jump engine.
    """
    start = (x, y, z)
    target = (8 - x, 8 - y, 8 - z)

    # no-op guard
    if start == target:
        return []

    # single direction vector
    dirs = np.array([[target[0] - start[0],
                      target[1] - start[1],
                      target[2] - start[2]]], dtype=np.int8)

    # hand off to the existing jump generator
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        position=start,
        directions=dirs,
        allow_capture=True,      # teleport may capture
        use_amd=True             # keep GPU path if available
    )
