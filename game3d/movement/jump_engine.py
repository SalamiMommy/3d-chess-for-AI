"""Jump Movement Engine - RAW MOVE GENERATION ONLY.

NO validation here. Just generates candidate moves.
Validation happens in generator.py.
"""

import numpy as np
from numba import njit, prange
from typing import TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, COLOR_DTYPE, SIZE, MAX_COORD_VALUE
)
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import UnifiedCacheManager


@njit(cache=True, fastmath=True, parallel=True)
def _generate_jump_targets(pos: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Generate jump target coordinates (no validation)."""
    targets = pos + offsets

    # Simple bounds filter
    valid_mask = ((targets[:, 0] >= 0) & (targets[:, 0] < SIZE) &
                  (targets[:, 1] >= 0) & (targets[:, 1] < SIZE) &
                  (targets[:, 2] >= 0) & (targets[:, 2] < SIZE))

    valid_targets = targets[valid_mask]

    # CRITICAL: Ensure we always return a valid array (even if empty)
    if valid_targets.shape[0] == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)

    return valid_targets


class JumpMovementEngine:
    """
    Raw jump move generation (Knight, King, etc.).

    NO validation - just generates candidate moves.
    Generator will validate them.
    """

    def __init__(self, cache_manager: 'UnifiedCacheManager'):
        self.cache_manager = cache_manager


    def generate_jump_moves(
            self,
            color: COLOR_DTYPE,
            pos: np.ndarray,
            directions: np.ndarray,          # (N,3) jump offsets
            allow_capture: bool = True
        ) -> np.ndarray:
            """Generate jump moves as numpy array [from_x, from_y, from_z, to_x, to_y, to_z]."""
            targets = _generate_jump_targets(pos, directions)

            if targets.shape[0] == 0:
                return np.empty((0, 6), dtype=COORD_DTYPE)

            flattened = self.cache_manager.occupancy_cache.get_flattened_occupancy()
            idxs = targets[:, 0] + SIZE * targets[:, 1] + SIZE * SIZE * targets[:, 2]
            occs = flattened[idxs]

            if allow_capture:
                valid_mask = (occs != color)
            else:
                valid_mask = (occs == 0)

            if not np.any(valid_mask):
                return np.empty((0, 6), dtype=COORD_DTYPE)

            valid_targets = targets[valid_mask]
            
            if valid_targets.shape[0] == 0:
                return np.empty((0, 6), dtype=COORD_DTYPE)

            # Construct (N, 6) array
            n_moves = valid_targets.shape[0]
            moves = np.empty((n_moves, 6), dtype=COORD_DTYPE)
            
            # Fill from_coord
            moves[:, 0] = pos[0]
            moves[:, 1] = pos[1]
            moves[:, 2] = pos[2]
            
            # Fill to_coord
            moves[:, 3:6] = valid_targets
            
            return moves



def get_jump_movement_generator(cache_manager: 'UnifiedCacheManager') -> JumpMovementEngine:
    """Backwards compatibility alias for JumpMovementEngine constructor."""
    return JumpMovementEngine(cache_manager)

# Update the __all__ export list
__all__ = ['JumpMovementEngine', 'get_jump_movement_generator']
