"""Exports (3,1) knight moves and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.knight31movement import generate_knight31_moves


@register(PieceType.KNIGHT31)
def knight31_dispatcher(state: GameState, x: int, y, z: int) -> List[Move]:
    return generate_knight31_moves(state, x, y, z)


__all__ = ['generate_knight31_moves']
