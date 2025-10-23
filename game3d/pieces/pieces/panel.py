# game3d/movement/pieces/panel.py
"""
Panel: teleport to any square on the same x OR y OR z plane plus king moves.
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
#  Panel directions (same x/y/z plane)                                       #
# --------------------------------------------------------------------------- #
_PANEL_DIRECTIONS = np.array([
    # X-axis lines (y,z constant)
    *[(dx, 0, 0) for dx in range(-8, 9) if dx != 0],
    # Y-axis lines (x,z constant)
    *[(0, dy, 0) for dy in range(-8, 9) if dy != 0],
    # Z-axis lines (x,y constant)
    *[(0, 0, dz) for dz in range(-8, 9) if dz != 0],
], dtype=np.int8)

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

def generate_panel_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Generate panel moves: king walks + plane teleports."""
    x, y, z = ensure_int_coords(x, y, z)

    # Combine directions and remove duplicates
    all_dirs = np.unique(np.vstack((_PANEL_DIRECTIONS, _KING_DIRECTIONS)), axis=0)

    # Generate all moves using jump movement
    jump_gen = get_integrated_jump_movement_generator(cache)
    moves = jump_gen.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=all_dirs,
        allow_capture=True,
    )

    return moves

@register(PieceType.PANEL)
def panel_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_panel_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_panel_moves"]
