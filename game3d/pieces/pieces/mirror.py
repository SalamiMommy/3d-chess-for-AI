# game3d/movement/pieces/mirror.py - FIXED
"""
Mirror-Teleporter – single jump to (8-x, 8-y, 8-z).
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.common.cache_utils import ensure_int_coords
from game3d.common.coord_utils import in_bounds

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def generate_mirror_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate mirror teleport moves."""
    x, y, z = ensure_int_coords(x, y, z)

    # Calculate mirror target
    target = (8 - x, 8 - y, 8 - z)

    # Check if target is valid and different from start
    if target == (x, y, z) or not in_bounds(target):
        return []

    # Check if target is occupied by friendly piece
    victim = cache_manager.get_piece(target)
    if victim is not None and victim.color == color:
        return []

    # Create direction array
    dx, dy, dz = target[0] - x, target[1] - y, target[2] - z
    directions = np.array([(dx, dy, dz)], dtype=np.int8)

    # FIXED: Use parameter name
    jump_gen = get_integrated_jump_movement_generator(cache_manager)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=directions,
        allow_capture=True,
    )

    return moves

@register(PieceType.MIRROR)
def mirror_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_mirror_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_mirror_moves"]
