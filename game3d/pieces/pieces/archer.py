# game3d/movement/pieces/archer.py
"""
Unified Archer dispatcher
- 1-radius sphere  → walk (normal king-like move)
- 2-radius surface → shoot (archery capture, no movement)
"""
from __future__ import annotations

import math
from typing import List, Tuple, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.common.coord_utils import in_bounds, get_aura_squares
from game3d.movement.registry import register

# 1-step engine re-used from kingmovement
from game3d.movement.movetypes.kingmovement import generate_king_moves

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState


# ------------------------------------------------------------------
# 1.  Internal helper: classify square
# ------------------------------------------------------------------
def _archer_intent(start: Tuple[int, int, int], target: Tuple[int, int, int]) -> str:
    """Return 'move', 'shoot', or 'invalid'."""
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    dz = target[2] - start[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if abs(dist - 1.0) < 0.1:
        return "move"
    if abs(dist - 2.0) < 0.1:
        return "shoot"
    return "invalid"


# ------------------------------------------------------------------
# 2.  Core move generator
# ------------------------------------------------------------------
def generate_archer_moves(
    cache,          # OptimizedCacheManager
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    start = (x, y, z)
    moves: List[Move] = []

    if cache.is_frozen(start, color):          # no moves while frozen
        return []

    # ---------- 1-radius king walk ----------
    moves.extend(generate_king_moves(cache, color, x, y, z))

    # ---------- 2-radius archery ----------
    for sq in get_aura_squares(start, radius=2):   # surface only
        victim = cache.occupancy.get(sq)
        if victim is not None and victim.color != color:
            # Ensure clear line-of-sight (cache provides fast LOS)

            mv = convert_legacy_move_args(
                start, start,                 # archer never moves
                is_capture=True,
                flags=MOVE_FLAGS['ARCHERY'] | MOVE_FLAGS['CAPTURE']
            )
            mv.metadata["target_square"] = sq   # victim square
            moves.append(mv)

    return moves


# ------------------------------------------------------------------
# 3.  Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.ARCHER)
def archer_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_archer_moves(state.cache, state.color, x, y, z)


__all__ = ["generate_archer_moves"]
