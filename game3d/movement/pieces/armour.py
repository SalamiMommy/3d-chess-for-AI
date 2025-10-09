# game3d/movement/piecemoves/armourmoves.py
"""Armour moves exactly like a King, but is tagged 'armoured'."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

__all__ = ["generate_armour_moves"]

def generate_armour_moves(
    cache: "OptimizedCacheManager",
    color: "Color",
    x: int, y: int, z: int
) -> List[Move]:
    """Re-use King generator; Piece already carries the armoured flag."""
    return generate_king_moves(cache, color, x, y, z)

@register(PieceType.ARMOUR)
def armour_move_dispatcher(state: "GameState", x: int, y: int, z: int) -> List[Move]:
    return generate_armour_moves(state.cache, state.color, x, y, z)
