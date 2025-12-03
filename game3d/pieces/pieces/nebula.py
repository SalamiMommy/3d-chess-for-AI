"""
Nebula piece implementation - teleport within radius-2 sphere (unbuffed) or radius-3 sphere (buffed).
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import COORD_DTYPE, COLOR_DTYPE, PieceType, RADIUS_2_OFFSETS, RADIUS_3_OFFSETS
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Unbuffed: All positions within radius 2 (excluding origin)
_NEBULA_DIRECTIONS = np.array([
    offset for offset in RADIUS_2_OFFSETS
    if not (offset[0] == 0 and offset[1] == 0 and offset[2] == 0)
], dtype=COORD_DTYPE)

# Buffed: All positions within radius 3 (excluding origin)
_BUFFED_NEBULA_DIRECTIONS = np.array([
    offset for offset in RADIUS_3_OFFSETS
    if not (offset[0] == 0 and offset[1] == 0 and offset[2] == 0)
], dtype=COORD_DTYPE)

def generate_nebula_moves(
    cache_manager: 'OptimizedCacheManager',
    color: COLOR_DTYPE,
    pos: np.ndarray
) -> np.ndarray:
    """Generate teleport moves within radius-2 sphere (unbuffed) or radius-3 sphere (buffed)."""

    # Get jump movement generator from cache
    jump_engine = get_jump_movement_generator()

    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos.astype(COORD_DTYPE),
        directions=_NEBULA_DIRECTIONS,
        allow_capture=True,
        piece_type=PieceType.NEBULA,
        buffed_directions=_BUFFED_NEBULA_DIRECTIONS
    )

@register(PieceType.NEBULA)
def nebula_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Nebula move dispatcher registered with piece type."""
    return generate_nebula_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_nebula_moves"]
