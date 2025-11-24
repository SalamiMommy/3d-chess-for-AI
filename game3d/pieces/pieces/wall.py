# game3d/movement/movetypes/wall.py - FULLY NUMPY-NATIVE
"""
Unified Wall movement + behind-capture logic - fully vectorized.
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.shared_types import Color, PieceType, Result, WALL
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.shared_types import COORD_DTYPE, SIZE, get_empty_coord_batch
from game3d.common.coord_utils import in_bounds_vectorized

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# Wall-specific movement vectors - orthogonal movement (6 directions) - numpy native
WALL_MOVEMENT_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

# 2×2×1 block geometry helpers
def _wall_squares_numpy(anchor: np.ndarray) -> np.ndarray:
    """Return the 4 squares occupied by a Wall anchored at anchor."""
    offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=COORD_DTYPE)
    return anchor.reshape(1, 3) + offsets

def _block_in_bounds_numpy(anchor: np.ndarray) -> bool:
    """True if the entire 2×2×1 block stays inside the board."""
    return bool(np.all((anchor >= 0) & (anchor < SIZE)) and np.all((anchor + [1, 1, 0] < SIZE)))

# Behind-mask builder (fully vectorized)
def _build_behind_mask_numpy(anchor: np.ndarray) -> np.ndarray:
    """
    Return every square that is **behind** the Wall anchored at *anchor*.
    Returns array of shape (N, 3) containing all behind squares.
    Fully vectorized implementation.
    """
    directions = WALL_MOVEMENT_VECTORS

    # Vectorized calculation of all possible positions in each direction
    max_steps = SIZE - 1
    step_range = np.arange(1, max_steps + 1, dtype=COORD_DTYPE)

    # Calculate all positions: anchor + direction * step for all combinations
    direction_expanded = directions[:, np.newaxis, :]
    step_expanded = step_range[np.newaxis, :, np.newaxis]
    anchor_expanded = anchor[np.newaxis, np.newaxis, :]

    all_positions = anchor_expanded + direction_expanded * step_expanded

    # Flatten and filter valid positions
    positions_2d = all_positions.reshape(-1, 3)

    # Vectorized bounds checking
    valid_mask = np.all((positions_2d >= 0) & (positions_2d < SIZE), axis=1)
    valid_positions = positions_2d[valid_mask]

    if valid_positions.shape[0] == 0:
        return get_empty_coord_batch()

    return valid_positions.astype(COORD_DTYPE)

def can_capture_wall_vectorized(attacker_sqs: np.ndarray, wall_anchors: np.ndarray) -> np.ndarray:
    """Check if attackers can capture walls - fully vectorized numpy operations."""
    if attacker_sqs.shape[0] == 0:
        return np.array([], dtype=bool)

    attacker_sqs = np.asarray(attacker_sqs, dtype=COORD_DTYPE)
    wall_anchors = np.asarray(wall_anchors, dtype=COORD_DTYPE)

    # Broadcast for all combinations: (num_attackers, num_walls, 3)
    diff = attacker_sqs[:, np.newaxis, :] - wall_anchors[np.newaxis, :, :]

    # Check alignment in each axis (2 coordinates must be zero, 1 must be non-zero)
    zero_mask = (diff == 0)
    zero_count = np.sum(zero_mask, axis=2)
    aligned_mask = (zero_count == 2)

    # For aligned pairs, check direction validity
    non_zero_mask = ~zero_mask
    non_zero_axis = np.where(non_zero_mask)

    if len(non_zero_axis[0]) > 0:
        attacker_indices, wall_indices, axis_indices = non_zero_axis
        diff_values = diff[attacker_indices, wall_indices, axis_indices]
        valid_direction_mask = diff_values > 0
        valid_directions = np.zeros_like(aligned_mask)
        valid_directions[attacker_indices, wall_indices] = valid_direction_mask

        return (aligned_mask & valid_directions).any(axis=1)

    return np.zeros(attacker_sqs.shape[0], dtype=bool)

def can_capture_wall_numpy(attacker_sq: np.ndarray, wall_anchor: np.ndarray) -> bool:
    """
    Return True if *attacker_sq* is **behind** the Wall anchored at *wall_anchor*.
    Numpy native version.
    """
    behind_mask = _build_behind_mask_numpy(wall_anchor)
    if len(behind_mask) == 0:
        return False

    # Check if attacker_sq matches any behind square
    return np.any(np.all(behind_mask == attacker_sq.reshape(1, 3), axis=1))

def generate_wall_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    anchor = pos.astype(COORD_DTYPE)

    if not _block_in_bounds_numpy(anchor):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Use wall-specific movement vectors
    jump_engine = get_jump_movement_generator(cache_manager)
    return jump_engine.generate_jump_moves(
        color=color,
        pos=anchor,
        directions=WALL_MOVEMENT_VECTORS,
        allow_capture=True,
        piece_type=PieceType.WALL
    )

@register(PieceType.WALL)
def wall_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_wall_moves(state.cache_manager, state.color, pos)

__all__ = ["WALL_MOVEMENT_VECTORS", "generate_wall_moves", "can_capture_wall_numpy", "can_capture_wall_vectorized"]
