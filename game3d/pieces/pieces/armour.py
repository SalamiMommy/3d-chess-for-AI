# game3d/movement/armour.py
"""
Unified ARMOUR dispatcher + protection logic.

ARMOUR pieces:
  - move exactly like a King (26 one-step directions)
  - are **immune to pawn captures**
  - can still be captured by any other piece (slider, bomb, etc.)
"""

from __future__ import annotations

from typing import List, Tuple, Set, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ------------------------------------------------------------------
# 1.  Movement – delegate to king engine
# ------------------------------------------------------------------
def generate_armour_moves(
    cache,          # OptimizedCacheManager
    color: Color,
    x: int, y: int, z: int
) -> List["Move"]:
    """Re-use the king generator; the Piece object carries the armoured flag."""
    return generate_king_moves(cache, color, x, y, z)


# ------------------------------------------------------------------
# 2.  Protection helpers – used by pawn generators
# ------------------------------------------------------------------
def is_armour_protected(sq: Tuple[int, int, int], cache) -> bool:
    """Return True if *sq* contains an ARMOUR piece (pawn capture blocked)."""
    piece = cache.occupancy.get(sq)
    return piece is not None and piece.ptype is PieceType.ARMOUR


def get_armoured_squares(cache, controller: Color) -> Set[Tuple[int, int, int]]:
    """All squares occupied by ARMOUR pieces of *controller*."""
    return {
        sq for sq, piece in cache.occupancy.iter_color(controller)
        if piece.ptype is PieceType.ARMOUR
    }


# ------------------------------------------------------------------
# 3.  Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.ARMOUR)
def armour_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List["Move"]:
    return generate_armour_moves(state.cache, state.color, x, y, z)


__all__ = ["generate_armour_moves", "is_armour_protected", "get_armoured_squares"]
