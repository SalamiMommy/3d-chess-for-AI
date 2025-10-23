# game3d/movement/pieces/bomb.py
"""
Unified Bomb generator – king steps + self-detonation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move, convert_legacy_move_args, MOVE_FLAGS
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.cache_utils import ensure_int_coords
from game3d.common.coord_utils import get_aura_squares

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# --------------------------------------------------------------------------- #
#  King directions (1-step moves)                                             #
# --------------------------------------------------------------------------- #
_KING_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

def generate_bomb_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all bomb moves: king walks + self-detonation."""
    x, y, z = ensure_int_coords(x, y, z)
    pos = (x, y, z)

    # 1. King walks using jump movement
    jump_gen = get_integrated_jump_movement_generator(cache)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=pos,
        directions=_KING_DIRECTIONS,
        allow_capture=True,
    )

    # 2. Self-detonation move (if it would affect enemies)
    if _detonate_would_affect_enemies(cache, pos, color):
        detonate_move = convert_legacy_move_args(
            from_coord=pos,
            to_coord=pos,
            flags=MOVE_FLAGS['SELF_DETONATE']
        )
        detonate_move.metadata["detonate"] = True
        moves.append(detonate_move)

    return moves

def _detonate_would_affect_enemies(
    cache: 'OptimizedCacheManager',
    center: Tuple[int, int, int],
    current_color: Color
) -> bool:
    """Check if detonation would affect any enemy pieces."""
    from game3d.attacks.check import _any_priest_alive

    for sq in get_aura_squares(center, radius=2):
        victim = cache.occupancy.get(sq)
        if victim is None or victim.color == current_color:
            continue
        if victim.ptype is PieceType.KING and _any_priest_alive(cache.board, victim.color):
            continue
        return True  # At least one enemy would be affected

    return False

@register(PieceType.BOMB)
def bomb_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_bomb_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_bomb_moves"]
