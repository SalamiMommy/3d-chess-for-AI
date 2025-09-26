# game3d/movement/piecemoves/bishopmoves.py
"""Exports bishop move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.bishopmovement import generate_bishop_moves
from game3d.movement.movepiece import Move

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_bishop_moves']


@register(PieceType.BISHOP)
def bishop_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_bishop_moves(state, x, y, z)
