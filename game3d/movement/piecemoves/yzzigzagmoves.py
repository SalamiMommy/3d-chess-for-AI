"""Exports YZ-Zig-Zag move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.yzzigzagmovement import generate_yz_zigzag_moves
from game3d.movement.movepiece import Move

__all__ = ['generate_yz_zigzag_moves']


@register(PieceType.YZZIGZAG)
def yz_zigzag_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_yz_zigzag_moves(state.cache, state.color, x, y, z)
