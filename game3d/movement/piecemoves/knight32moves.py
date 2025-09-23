"""Exports (3,2) knight moves and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.knight32movement import generate_knight32_moves


@register(PieceType._KNIGHT32)
def knight32_dispatcher(state: GameState, x: int, y, z: int) -> List[Move]:
    return generate_knight32_moves(state, x, y, z)


__all__ = ['generate_knight32_moves']
