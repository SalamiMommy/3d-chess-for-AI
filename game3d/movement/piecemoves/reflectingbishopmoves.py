"""Exports Reflecting-Bishop moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.reflectingbishopmovement import generate_reflecting_bishop_moves
from game3d.movement.movepiece import Move

@register(PieceType.REFLECTOR)
def reflecting_bishop_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_reflecting_bishop_moves(state.cache, state.color, x, y, z)  # ← PASS CACHE

__all__ = ['generate_reflecting_bishop_moves']
