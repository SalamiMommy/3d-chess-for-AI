"""Exports (3,2) knight moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.knight32movement import generate_knight32_moves
from game3d.movement.movepiece import Move

@register(PieceType.KNIGHT32)
def knight32_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_knight32_moves(state, x, y, z)


__all__ = ['generate_knight32_moves']
