"""
Echo piece implementation - 1-sphere surface projection with ±2 axis offset.

The Echo piece moves along a 1-sphere surface by projecting from 6 cardinal anchor points
(offset by 2 spaces) and adding 26 radius-1 bubble offsets.
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, RADIUS_1_OFFSETS, PieceType
)
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState
    from game3d.movement.movepiece import Move

# Echo piece-specific movement vectors (numpy arrays)
# 6 cardinal anchors at offset 2 (unbuffed)
_ANCHORS = np.array([
    [-2, 0, 0], [2, 0, 0],
    [0, -2, 0], [0, 2, 0],
    [0, 0, -2], [0, 0, 2]
], dtype=COORD_DTYPE)

# 6 cardinal anchors at offset 3 (buffed - 1 space further)
_BUFFED_ANCHORS = np.array([
    [-3, 0, 0], [3, 0, 0],
    [0, -3, 0], [0, 3, 0],
    [0, 0, -3], [0, 0, 3]
], dtype=COORD_DTYPE)

# 26 radius-1 bubble offsets
_BUBBLE = RADIUS_1_OFFSETS.copy()

# 156 raw jump vectors (anchors + bubbles) - unbuffed
_ECHO_DIRECTIONS = (_ANCHORS[:, None, :] + _BUBBLE[None, :, :]).reshape(-1, 3)

# 156 raw jump vectors (buffed anchors + bubbles) - buffed
_BUFFED_ECHO_DIRECTIONS = (_BUFFED_ANCHORS[:, None, :] + _BUBBLE[None, :, :]).reshape(-1, 3)

def generate_echo_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate echo piece movement vectors using vectorized operations.

    The Echo moves along a 1-sphere surface by projecting from 6 cardinal anchor points
    (offset by 2 unbuffed, 3 when buffed) and adding radius-1 bubble offsets for complex movement patterns.

    Args:
        cache_manager: Cache manager for board state access
        color: Piece color (WHITE/BLACK)
        pos: Starting position as numpy array

    Returns:
        Array of valid movement vectors
    """
    start = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if start.ndim == 1:
        start = start.reshape(1, 3)

    # Generate jump movements through cache manager
    jump_engine = get_jump_movement_generator()
    return jump_engine.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=_ECHO_DIRECTIONS,
        allow_capture=True,
        piece_type=PieceType.ECHO,
        buffed_directions=_BUFFED_ECHO_DIRECTIONS
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
