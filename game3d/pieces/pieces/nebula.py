"""
Nebula piece implementation - teleport to any position within radius-3 sphere.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.shared_types import COORD_DTYPE, COLOR_DTYPE, PieceType
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def generate_nebula_moves(
    cache_manager: 'OptimizedCacheManager',
    color: COLOR_DTYPE,
    pos: np.ndarray
) -> np.ndarray:
    """Generate teleport moves within radius-3 sphere."""

    # Get jump movement generator from cache
    jump_engine = get_jump_movement_generator()

    # All positions at Manhattan distance 3 (simplified from original)
    directions = np.array([
        # Axial movements
        (-3, 0, 0), (3, 0, 0), (0, -3, 0), (0, 3, 0), (0, 0, -3), (0, 0, 3),
        # L-shaped movements (Manhattan distance 3)
        (-2, -1, 0), (-2, 0, -1), (-2, 0, 1), (-2, 1, 0),
        (2, -1, 0), (2, 0, -1), (2, 0, 1), (2, 1, 0),
        (-1, -2, 0), (-1, 0, -2), (-1, 0, 2), (-1, 2, 0),
        (1, -2, 0), (1, 0, -2), (1, 0, 2), (1, 2, 0),
        (0, -2, -1), (0, -2, 1), (0, -1, -2), (0, -1, 2),
        (0, 1, -2), (0, 1, 2), (0, 2, -1), (0, 2, 1),
        # 3D diagonal-ish movements
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ], dtype=COORD_DTYPE)

    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=pos.astype(COORD_DTYPE),
        directions=directions,
        allow_capture=True,
        piece_type=PieceType.NEBULA
    )

@register(PieceType.NEBULA)
def nebula_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Nebula move dispatcher registered with piece type."""
    return generate_nebula_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_nebula_moves"]
