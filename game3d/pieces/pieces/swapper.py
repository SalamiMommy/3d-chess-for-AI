# game3d/movement/pieces/swapper.py
"""
Swapper == King-steps ∪ friendly-swap teleport
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# --------------------------------------------------------------------------- #
#  King directions (1-step moves)                                             #
# --------------------------------------------------------------------------- #
_KING_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

def generate_swapper_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate swapper moves: king walks + friendly swaps."""
    x, y, z = ensure_int_coords(x, y, z)

    jump_gen = get_integrated_jump_movement_generator(cache)
    moves = []

    # 1. King walks
    king_moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=_KING_DIRECTIONS,
        allow_capture=True,
    )
    moves.extend(king_moves)

    # 2. Friendly swaps
    swap_dirs = _get_friendly_swap_directions(cache, color, x, y, z)
    if len(swap_dirs) > 0:
        swap_moves = jump_gen.generate_jump_moves(
            color=color,
            pos=(x, y, z),
            directions=swap_dirs,
            allow_capture=False,  # Swaps don't capture
        )

        # Mark as swaps
        for move in swap_moves:
            move.metadata["is_swap"] = True

        moves.extend(swap_moves)

    return moves

def _get_friendly_swap_directions(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> np.ndarray:
    """Get directions to all friendly pieces (excluding self)."""
    start = np.array([x, y, z], dtype=np.int16)

    friendly_coords = []
    for coord, piece in cache.occupancy.iter_color(color):
        if coord != (x, y, z):
            friendly_coords.append(coord)

    if not friendly_coords:
        return np.empty((0, 3), dtype=np.int8)

    friendly_arr = np.array(friendly_coords, dtype=np.int16)
    directions = (friendly_arr - start).astype(np.int8)

    return directions

@register(PieceType.SWAPPER)
def swapper_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_swapper_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_swapper_moves"]
