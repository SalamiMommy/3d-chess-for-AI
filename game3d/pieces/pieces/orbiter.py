"""Orbiter piece - Manhattan-distance movement with 66 possible jumps."""
from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, PieceType, ORBITER, ORBITER_MANHATTAN_DISTANCE
)
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# Precomputed movement vectors - 66 positions at exactly 4 Manhattan distance
_ORBITAL_DIRS = np.array([
    (dx, dy, dz)
    for dx in range(-ORBITER_MANHATTAN_DISTANCE, ORBITER_MANHATTAN_DISTANCE + 1)
    for dy in range(-ORBITER_MANHATTAN_DISTANCE, ORBITER_MANHATTAN_DISTANCE + 1)
    for dz in range(-ORBITER_MANHATTAN_DISTANCE, ORBITER_MANHATTAN_DISTANCE + 1)
    if abs(dx) + abs(dy) + abs(dz) == ORBITER_MANHATTAN_DISTANCE
], dtype=COORD_DTYPE)

def generate_orbital_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray  # ✅ Changed: accept numpy array directly
) -> np.ndarray:
    """Generate all valid Orbiter moves - jumps to positions exactly 4 Manhattan away."""
    return get_jump_movement_generator(cache_manager).generate_jump_moves(
        color=color, pos=pos, directions=_ORBITAL_DIRS
    )

@register(PieceType.ORBITER)
def orbital_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Registered move dispatcher for Orbiter pieces."""
    return generate_orbital_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_orbital_moves"]
