# game3d/movement/piecemoves/rookmoves.py
"""Exports rook move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.rookmovement import generate_rook_moves
from game3d.movement.movepiece import Move

__all__ = ['generate_rook_moves']


@register(PieceType.ROOK)
def rook_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_rook_moves(state, x, y, z)
