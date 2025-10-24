"""
Rook — orthogonal slider rays.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.slidermovement import generate_moves
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# 6 orthogonal directions (±X, ±Y, ±Z)
_ROOK_DIRS = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
], dtype=np.int8)

def generate_rook_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Consistent parameter name
    color: Color,
    x: int, y: int, z: int,
    max_steps: int = 8
) -> List[Move]:
    """Orthogonal slider; steps capped by max_steps."""
    return generate_moves(
        piece_type='rook',
        pos=(x, y, z),
        color=color,
        max_distance=max_steps,
        directions=_ROOK_DIRS,
        cache_manager=cache_manager,  # FIXED: Use parameter
    )

@register(PieceType.ROOK)
def rook_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_rook_moves(state.cache_manager, state.color, x, y, z, 8)  # FIXED: Use cache_manager

__all__ = ['generate_rook_moves']
