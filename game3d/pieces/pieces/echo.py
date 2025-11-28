"""
Echo piece implementation - 2-sphere surface projection with ±3 axis movement.

The Echo piece moves along a 2-sphere surface by projecting from 8 anchor points
(±2, ±2, ±2) and adding 32 radius-2 bubble offsets, creating 256 total movement vectors.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, RADIUS_2_OFFSETS, PieceType
)
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState
    from game3d.movement.movepiece import Move

# Echo piece-specific movement vectors (numpy arrays)
# 8 anchor offsets (±2, ±2, ±2)
_ANCHORS = np.array([
    [-2, -2, -2], [-2, -2, 2], [-2, 2, -2], [-2, 2, 2],
    [2, -2, -2], [2, -2, 2], [2, 2, -2], [2, 2, 2]
], dtype=COORD_DTYPE)

# 32 radius-2 bubble offsets from shared types
_BUBBLE = RADIUS_2_OFFSETS.copy()

# 256 raw jump vectors (anchors + bubbles)
_ECHO_DIRECTIONS = (_ANCHORS[:, None, :] + _BUBBLE[None, :, :]).reshape(-1, 3)

def generate_echo_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate echo piece movement vectors using vectorized operations.

    The Echo moves along a 2-sphere surface by projecting from anchor points
    and adding radius-2 bubble offsets for complex movement patterns.

    Args:
        cache_manager: Cache manager for board state access
        color: Piece color (WHITE/BLACK)
        pos: Starting position as numpy array

    Returns:
        Array of valid movement vectors
    """
    start = pos.astype(COORD_DTYPE).ravel()

    # Generate jump movements through cache manager
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=_ECHO_DIRECTIONS,
        allow_capture=True,
        piece_type=PieceType.ECHO
    )

@register(PieceType.ECHO)
def echo_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Echo piece movement dispatcher for game state integration.

    Args:
        state: Game state with cache manager access
        pos: Current position as numpy array

    Returns:
        Array of valid echo movements
    """
    return generate_echo_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_echo_moves"]
