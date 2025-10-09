# game3d/movement/movetypes/wall.py
"""
Unified Wall movement generator + behind-capture logic.

Wall pieces:
  - occupy a **2×2×1** block of squares;
  - move exactly like a King (one step in any of the 26 directions)
    – but the **entire block** must remain inside the 9×9×9 board;
  - **cannot capture** (all its moves have `is_capture = False`);
  - can **only be captured from behind** (sliders or Bomb detonation);
  - the “behind” mask is read straight from the cache manager.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, convert_legacy_move_args
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ------------------------------------------------------------------
# 26 one-step directions (same as King)
# ------------------------------------------------------------------
KING_DIRECTIONS_3D = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)

# ------------------------------------------------------------------
# Helpers for 2×2×1 block geometry
# ------------------------------------------------------------------
def _wall_squares(anchor: Tuple[int, int, int]) -> list[Tuple[int, int, int]]:
    """Return the 4 squares occupied by a Wall anchored at (x,y,z)."""
    x, y, z = anchor
    return [
        (x, y, z),
        (x + 1, y, z),
        (x, y + 1, z),
        (x + 1, y + 1, z),
    ]

def _block_in_bounds(anchor: Tuple[int, int, int]) -> bool:
    """True if the entire 2×2×1 block stays inside the 9×9×9 board."""
    x, y, z = anchor
    return x + 1 < 9 and y + 1 < 9 and 0 <= z < 9

# ------------------------------------------------------------------
# Public generator – used by dispatcher and AI
# ------------------------------------------------------------------
def generate_wall_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """All legal Wall moves from its anchor square (x,y,z)."""
    anchor = (x, y, z)
    if not _block_in_bounds(anchor):  # safety check
        return []

    gen = get_integrated_jump_movement_generator(state.cache)
    moves: List[Move] = []

    for dx, dy, dz in KING_DIRECTIONS_3D:
        new_anchor = (x + dx, y + dy, z + dz)
        if not _block_in_bounds(new_anchor):
            continue  # part of the block would fall off the board

        # Build the move – **never a capture**
        moves.append(convert_legacy_move_args(
            from_coord=anchor,
            to_coord=new_anchor,
            is_capture=False
        ))

    return moves

# ------------------------------------------------------------------
# Behind-capture validation (used by the cache manager)
# ------------------------------------------------------------------
def can_capture_wall(attacker_sq: Tuple[int, int, int],
                     wall_anchor: Tuple[int, int, int],
                     controller: Color,
                     cache_manager) -> bool:
    """
    Return True if *attacker_sq* is **behind** the Wall anchored at *wall_anchor*.
    This is called by the cache manager when a slider or Bomb tries to capture.
    """
    behind = cache_manager.wall_behind_squares(controller).get(wall_anchor)
    return behind is not None and attacker_sq in behind

# ------------------------------------------------------------------
# Dispatcher registration (old name kept)
# ------------------------------------------------------------------
@register(PieceType.WALL)
def wall_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_wall_moves(state, x, y, z)

# ------------------------------------------------------------------
# Backward compatibility exports
# ------------------------------------------------------------------
__all__ = ["generate_wall_moves", "can_capture_wall"]
