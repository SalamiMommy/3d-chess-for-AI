# game3d/movement/movetypes/wall.py - UPDATED
"""
Unified Wall movement + behind-capture logic.
"""

from __future__ import annotations

from typing import List, Tuple, Set, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.common.coord_utils import in_bounds, add_coords
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# 2×2×1 block geometry helpers
def _wall_squares(anchor: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Return the 4 squares occupied by a Wall anchored at (x,y,z)."""
    x, y, z = anchor
    return [(x, y, z), (x + 1, y, z), (x, y + 1, z), (x + 1, y + 1, z)]

def _block_in_bounds(anchor: Tuple[int, int, int]) -> bool:
    """True if the entire 2×2×1 block stays inside the board."""
    x, y, z = anchor
    return 0 <= x + 1 < 9 and 0 <= y + 1 < 9 and 0 <= z < 9

# Behind-mask builder (used for capture validation)
def _build_behind_mask(anchor: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """
    Return every square that is **behind** the Wall anchored at *anchor*.
    """
    cx, cy, cz = anchor
    behind: Set[Tuple[int, int, int]] = set()

    # 4 axial directions that point *away* from the block centre
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0),
                       (0, 1, 0), (0, -1, 0),
                       (0, 0, 1), (0, 0, -1)):
        for step in range(1, 9):
            sq = add_coords(anchor, (dx * step, dy * step, dz * step))
            if not in_bounds(sq):
                break
            behind.add(sq)
    return behind

# Batch version for multiple attacker squares
def can_capture_wall_batch(attacker_sqs: List, wall_anchor: Tuple[int, int, int]) -> List[bool]:
    """
    Batch version of can_capture_wall for multiple attacker squares.
    Returns a list of booleans indicating if each attacker square can capture the wall.

    Handles both list of tuples and list of lists as input.
    """
    behind_mask = _build_behind_mask(wall_anchor)

    results = []
    for attacker_sq in attacker_sqs:
        # Convert to tuple if it's a list or numpy array
        if isinstance(attacker_sq, (list, np.ndarray)):
            attacker_tuple = tuple(attacker_sq)
        else:
            attacker_tuple = attacker_sq
        results.append(attacker_tuple in behind_mask)

    return results

# Public generator – king steps for the whole block
def generate_wall_moves(
    cache_manager: 'OptimizedCacheManager',  # FIXED: Added parameter
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    anchor = (x, y, z)
    if not _block_in_bounds(anchor):
        return []

    # Re-use king engine for the anchor; block validity is checked later
    # FIXED: Use parameter name
    return generate_king_moves(cache_manager, color, x, y, z)

# Behind-capture validator (called by cache manager)
def can_capture_wall(attacker_sq: Tuple[int, int, int],
                     wall_anchor: Tuple[int, int, int]) -> bool:
    """
    Return True if *attacker_sq* is **behind** the Wall anchored at *wall_anchor*.
    """
    return attacker_sq in _build_behind_mask(wall_anchor)

# Dispatcher registration
@register(PieceType.WALL)
def wall_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    # FIXED: Use cache_manager property and pass to generator
    return generate_wall_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_wall_moves", "can_capture_wall", "can_capture_wall_batch"]
