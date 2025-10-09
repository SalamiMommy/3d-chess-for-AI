"""
Unified Archer dispatcher + move generator
1-radius sphere  → walk (normal move)
2-radius surface → shoot (archery capture, no movement)
"""
from __future__ import annotations

import math
from typing import List, Tuple, TYPE_CHECKING

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.registry import register

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# ------------------------------------------------------------------
# 1.  Internal helper: classify square
# ------------------------------------------------------------------
def _archer_intent(start: Tuple[int, int, int], target: Tuple[int, int, int]) -> str:
    """Return 'move', 'shoot', or 'invalid'."""
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    dz = target[2] - start[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if abs(dist - 1.0) < 0.1:
        return "move"
    if abs(dist - 2.0) < 0.1:
        return "shoot"
    return "invalid"

# ------------------------------------------------------------------
# 2.  Core move generator
# ------------------------------------------------------------------
def generate_archer_moves(
    cache: OptimizedCacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    start = (x, y, z)
    moves: List[Move] = []

    # Check if archer is affected by any effects
    if cache.is_frozen(start, color):
        return []  # No moves if frozen

    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            for dz in (-2, -1, 0, 1, 2):
                if dx == dy == dz == 0:
                    continue
                target = (x + dx, y + dy, z + dz)
                if not in_bounds(target):
                    continue

                intent = _archer_intent(start, target)
                if intent == "move":
                    # Use cache.occupancy instead of cache.piece_cache
                    victim = cache.occupancy.get(target)
                    is_cap = victim is not None and victim.color != color
                    moves.append(Move.create_simple(start, target, is_capture=is_cap))

                elif intent == "shoot":
                    # Use cache.occupancy instead of cache.piece_cache
                    victim = cache.occupancy.get(target)
                    if victim and victim.color != color:
                        # Check if target is valid for archery attack
                        if cache.is_valid_archery_attack(target, color):
                            mv = Move(start, start,
                                      flags=MOVE_FLAGS['ARCHERY'] | MOVE_FLAGS['CAPTURE'],
                                      captured_piece=victim.ptype.value)
                            mv.metadata["target_square"] = target
                            moves.append(mv)
    return moves

# ------------------------------------------------------------------
# 3.  Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.ARCHER)
def archer_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_archer_moves(state.cache, state.color, x, y, z)

# ------------------------------------------------------------------
# 4.  Public alias (kept for backward compatibility)
# ------------------------------------------------------------------
__all__ = ["generate_archer_moves"]
