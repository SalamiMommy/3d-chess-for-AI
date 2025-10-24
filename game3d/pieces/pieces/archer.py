# game3d/movement/pieces/archer.py - FIXED
"""
Unified Archer dispatcher
- 1-radius sphere  → walk (normal king-like move)
- 2-radius surface → shoot (archery capture, no movement)
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move, convert_legacy_move_args, MOVE_FLAGS
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# King directions (1-step moves)
_KING_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# Archery directions (2-radius surface only)
_ARCHERY_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if dx*dx + dy*dy + dz*dz == 4  # Only surface of sphere radius 2
], dtype=np.int8)

def generate_archer_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate all archer moves: king walks + archery shots."""
    x, y, z = ensure_int_coords(x, y, z)

    if cache_manager.is_frozen((x, y, z), color):
        return []

    moves = []

    # 1. King walks using jump movement - FIXED: Use parameter name
    jump_gen = get_integrated_jump_movement_generator(cache_manager)  # FIXED: cache_manager
    king_moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=_KING_DIRECTIONS,
        allow_capture=True,
    )
    moves.extend(king_moves)

    # 2. Archery shots (2-radius surface capture only)
    for dx, dy, dz in _ARCHERY_DIRECTIONS:
        tx, ty, tz = x + dx, y + dy, z + dz

        # Check bounds
        if not (0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9):
            continue

        # Check for enemy piece at target using public API
        victim = cache_manager.get_piece((tx, ty, tz))
        if victim is not None and victim.color != color:
            # Create archery move (archer doesn't move, just captures)
            archery_move = convert_legacy_move_args(
                from_coord=(x, y, z),
                to_coord=(x, y, z),  # Archer stays in place
                is_capture=True,
                flags=MOVE_FLAGS['ARCHERY'] | MOVE_FLAGS['CAPTURE']
            )
            archery_move.metadata["target_square"] = (tx, ty, tz)
            archery_move.metadata["archery_shot"] = True

            moves.append(archery_move)

    return moves

@register(PieceType.ARCHER)
def archer_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_archer_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_archer_moves"]
