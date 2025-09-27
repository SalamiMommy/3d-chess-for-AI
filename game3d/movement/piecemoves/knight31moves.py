"""Exports (3,1) knight moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.knight31movement import generate_knight31_moves
from game3d.movement.movepiece import Move

@register(PieceType.KNIGHT31)
def knight31_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_knight31_moves(state.board, state.color, x, y, z)


__all__ = ['generate_knight31_moves']
