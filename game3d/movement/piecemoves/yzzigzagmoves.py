"""Exports YZ-Zig-Zag slider moves (kingless) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.yzzigzagmovement import generate_yz_zigzag_moves
from game3d.movement.movepiece import Move

@register(PieceType.YZZIGZAG)
def yz_zigzag_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    """Registered dispatcher for YZ-Zig-Zag slider."""
    return generate_yz_zigzag_moves(state, x, y, z)


__all__ = ['generate_yz_zigzag_moves']
