"""Exports XZ-Zig-Zag move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.xzzigzagmovement import generate_xz_zigzag_moves
from game3d.movement.movepiece import Move

__all__ = ['generate_xz_zigzag_moves']


@register(PieceType.XZZIGZAG)
def xz_zigzag_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_xz_zigzag_moves(state.cache, state.color, x, y, z)
