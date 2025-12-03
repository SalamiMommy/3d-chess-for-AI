"""Orbiter piece - Euclidean sphere surface movement (radius 3 unbuffed, 4 buffed)."""
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

# Unbuffed: Radius 3 Euclidean sphere (surface only)
# r^2 = 9, accepting points where squared distance is close to 9
# We'll accept 8 <= d^2 <= 11 to get a good approximation of the sphere surface
_ORBITAL_DIRS = np.array([
    (dx, dy, dz)
    for dx in range(-4, 5)
    for dy in range(-4, 5)
    for dz in range(-4, 5)
    if 8 <= (dx*dx + dy*dy + dz*dz) <= 11
], dtype=COORD_DTYPE)

# Buffed: Radius 4 Euclidean sphere (surface only)
# r^2 = 16, accepting points where squared distance is close to 16
# We'll accept 14 <= d^2 <= 18 to get a good approximation of the sphere surface
_BUFFED_ORBITAL_DIRS = np.array([
    (dx, dy, dz)
    for dx in range(-5, 6)
    for dy in range(-5, 6)
    for dz in range(-5, 6)
    if 14 <= (dx*dx + dy*dy + dz*dz) <= 18
], dtype=COORD_DTYPE)

def generate_orbital_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray  # ✅ Changed: accept numpy array directly
) -> np.ndarray:
    """Generate all valid Orbiter moves - jumps to sphere surface (radius 3 unbuffed, 4 buffed)."""
    return get_jump_movement_generator().generate_jump_moves(
        cache_manager=cache_manager,
        color=color, pos=pos, 
        directions=_ORBITAL_DIRS, 
        piece_type=PieceType.ORBITER,
        buffed_directions=_BUFFED_ORBITAL_DIRS
    )

@register(PieceType.ORBITER)
def orbital_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Registered move dispatcher for Orbiter pieces."""
    return generate_orbital_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_orbital_moves"]
