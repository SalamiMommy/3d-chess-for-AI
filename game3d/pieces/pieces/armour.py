# game3d/movement/armour.py
"""
Unified ARMOUR dispatcher + protection logic.
"""

from __future__ import annotations

from typing import List, Tuple, Set, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.common.cache_utils import get_occupancy_safe, ensure_int_coords

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# ------------------------------------------------------------------
# 1.  Movement – delegate to king engine
# ------------------------------------------------------------------
def generate_armour_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List["Move"]:
    """Re-use the king generator; the Piece object carries the armoured flag."""
    x, y, z = ensure_int_coords(x, y, z)
    return generate_king_moves(cache, color, x, y, z)

# ------------------------------------------------------------------
# 2.  Protection helpers – used by pawn generators
# ------------------------------------------------------------------
def is_armour_protected(sq: Tuple[int, int, int], cache: 'OptimizedCacheManager') -> bool:
    """Return True if *sq* contains an ARMOUR piece (pawn capture blocked)."""
    piece = get_occupancy_safe(cache, sq)
    return piece is not None and piece.ptype is PieceType.ARMOUR

def get_armoured_squares(cache: 'OptimizedCacheManager', controller: Color) -> Set[Tuple[int, int, int]]:
    """All squares occupied by ARMOUR pieces of *controller*."""
    return {
        sq for sq, piece in cache.get_pieces_of_color(controller)
        if piece.ptype is PieceType.ARMOUR
    }

# ------------------------------------------------------------------
# 3.  Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.ARMOUR)
def armour_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List["Move"]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_armour_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_armour_moves", "is_armour_protected", "get_armoured_squares"]
