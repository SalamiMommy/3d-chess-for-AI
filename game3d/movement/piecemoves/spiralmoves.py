"""Exports Spiral moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.spiralmovement import generate_spiral_moves
from game3d.movement.movepiece import Move

@register(PieceType.SPIRAL)
def spiral_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_spiral_moves(state.cache, state.color, x, y, z)


__all__ = ['generate_spiral_moves']
