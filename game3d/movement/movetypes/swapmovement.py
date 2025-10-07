# swapmovement.py
# game3d/movement/movetypes/swapmovement.py
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

def generate_swapper_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Swap with any friendly piece using cached targets and jump engine."""
    start = (x, y, z)

    # Initialize cache attributes if not present
    if not hasattr(cache, '_swap_targets'):
        cache._swap_targets = {Color.WHITE: set(), Color.BLACK: set()}
        cache._swap_targets_dirty = {Color.WHITE: True, Color.BLACK: True}

    # Recalculate targets if cache is dirty
    if cache._swap_targets_dirty[color]:
        _update_swap_targets(cache, color)
        cache._swap_targets_dirty[color] = False

    # Get cached targets for this color
    targets = cache._swap_targets[color]

    # Remove the starting position if present
    if start in targets:
        targets_without_start = targets - {start}
    else:
        targets_without_start = targets

    if not targets_without_start:
        return []

    # Convert to numpy array for direction computation
    targets_array = np.array(list(targets_without_start), dtype=np.int16)
    start_array = np.array(start, dtype=np.int16)
    directions = targets_array - start_array

    # Hand off to jump movement generator
    gen = get_integrated_jump_movement_generator(cache)
    swap_moves = gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=directions,
        allow_capture=False,
    )

    # Mark as swap
    for m in swap_moves:
        m.is_swap = True
    return swap_moves

def _update_swap_targets(cache: CacheManager, color: Color) -> None:
    """
    Update the cached swap targets (friendly pieces) for the given color.
    """
    targets = set()

    # Get all friendly pieces for color
    if hasattr(cache.piece_cache, "iter_color"):
        targets = {coord for coord, _ in cache.piece_cache.iter_color(color)}

    cache._swap_targets[color] = targets
