# game3d/movement/piecemoves/hivemoves.py
"""Hive move generator – king-like single steps + multi-move turn helpers."""

from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movepiece import Move
from game3d.movement.movetypes.jumpmovement import (
    get_integrated_jump_movement_generator,
)
from game3d.movement.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# ------------------------------------------------------------------
#  Hive moves exactly one step in any of the 26 directions (like a King)
# ------------------------------------------------------------------
HIVE_DIRECTIONS_3D = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ------------------------------------------------------------------
#  Single-step generator – reused by the dispatcher
# ------------------------------------------------------------------
def generate_hive_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int,
) -> List[Move]:
    """Return every **one-step** Hive move from the given coordinate."""
    x, y, z = ensure_int_coords(x, y, z)
    engine = get_integrated_jump_movement_generator(cache)
    return engine.generate_jump_moves(
        color=color,
        pos=(x, y, z),
        directions=HIVE_DIRECTIONS_3D,
        allow_capture=True,
        piece_name="hive",
    )

# ------------------------------------------------------------------
#  Dispatcher registered for PieceType.HIVE
# ------------------------------------------------------------------
@register(PieceType.HIVE)
def hive_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_hive_moves(state.cache, state.color, x, y, z)

# ------------------------------------------------------------------
#  Multi-Hive turn helpers (unchanged behaviour)
# ------------------------------------------------------------------
def get_movable_hives(state: 'GameState', color: Color) -> List[Move]:
    """All single-step Hive moves still legal for *color* this turn."""
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves_for_piece

    hives: List[Move] = []
    for coord, piece in state.cache_manager.get_pieces_of_color(color):
        if piece.ptype is PieceType.HIVE:
            hives.extend(generate_pseudo_legal_moves_for_piece(state, coord))
    return hives

def apply_multi_hive_move(state: 'GameState', move: Move) -> 'GameState':
    """
    Apply *one* Hive move and return the new state **without** flipping
    the side-to-move flag.  The same player may continue moving Hives.
    """
    new_state = state.make_move(move)
    # Undo the colour flip that make_move performed
    object.__setattr__(new_state, "color", state.color)
    # Force re-computation of legal moves for the next pick
    new_state._clear_caches()
    return new_state

# ------------------------------------------------------------------
#  Keep old public names for 100 % backward compatibility
# ------------------------------------------------------------------
generate_king_moves = generate_hive_moves  # historical alias

__all__ = [
    "generate_hive_moves",
    "generate_king_moves",
    "get_movable_hives",
    "apply_multi_hive_move",
]
