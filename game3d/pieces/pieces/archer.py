# game3d/movement/pieces/archer.py - FULLY NUMPY-NATIVE
"""
Unified Archer dispatcher
- 1-radius sphere  → walk (normal king-like move)
- 2-radius surface → shoot (archery capture, no movement)
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from game3d.common.shared_types import *
from game3d.common.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# King directions (1-step moves) - optimized numpy construction
_KING_DIRECTIONS = np.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T.astype(COORD_DTYPE)
# Remove origin
origin = np.array([0, 0, 0], dtype=COORD_DTYPE)
_KING_DIRECTIONS = _KING_DIRECTIONS[~np.all(_KING_DIRECTIONS == origin, axis=1)]

# Archery directions (2-radius surface only) - optimized numpy construction
coords = np.mgrid[-2:3, -2:3, -2:3].reshape(3, -1).T
distances = np.sum(coords * coords, axis=1)
_ARCHERY_DIRECTIONS = coords[distances == 4].astype(COORD_DTYPE)

def generate_archer_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> list[Move]:
    """Generate all archer moves: king walks + archery shots."""
    start = pos.astype(COORD_DTYPE)

    moves_list = []

    # 1. King walks using jump movement (already vectorized)
    jump_gen = get_jump_movement_generator()
    king_moves = jump_gen.generate_jump_moves(
        cache_manager=cache_manager,
        color=color,
        pos=start,
        directions=_KING_DIRECTIONS,
        allow_capture=True,
    )
    if king_moves.size > 0:
        moves_list.append(king_moves)

    # 2. Archery shots (2-radius surface capture only) - FULLY VECTORIZED

    # Generate all possible archery targets at once using broadcasting
    # _ARCHERY_DIRECTIONS is (N, 3), start is (3,), result is (N, 3)
    targets = start + _ARCHERY_DIRECTIONS

    # Vectorized bounds check - filters out-of-board targets
    valid_mask = in_bounds_vectorized(targets)
    valid_targets = targets[valid_mask]

    if valid_targets.shape[0] > 0:
        # Vectorized occupancy check using flattened cache (same pattern as jump_engine.py)
        flattened = cache_manager.occupancy_cache.get_flattened_occupancy()
        idxs = valid_targets[:, 0] + SIZE * valid_targets[:, 1] + SIZE * SIZE * valid_targets[:, 2]
        occs = flattened[idxs]

        # Filter for enemy pieces only (occupied by opponent)
        enemy_mask = (occs != 0) & (occs != color)
        enemy_targets = valid_targets[enemy_mask]

        if enemy_targets.shape[0] > 0:
            n_shots = enemy_targets.shape[0]
            shot_moves = np.empty((n_shots, 6), dtype=COORD_DTYPE)
            shot_moves[:, 0:3] = start
            shot_moves[:, 3:6] = enemy_targets
            moves_list.append(shot_moves)

    if not moves_list:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    return np.concatenate(moves_list, axis=0)

@register(PieceType.ARCHER)
def archer_move_dispatcher(state: 'GameState', pos: np.ndarray) -> list[Move]:
    return generate_archer_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_archer_moves"]
