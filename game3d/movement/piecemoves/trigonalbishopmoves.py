# game3d/movement/piecemoves/trigonalbishopmoves.py
"""Exports trigonal bishop move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.trigonalbishopmovement import generate_trigonal_bishop_moves
from game3d.movement.movepiece import Move

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_trigonal_bishop_moves']


@register(PieceType.TRIGONALBISHOP)
def trigonal_bishop_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_trigonal_bishop_moves(state.board, state.color, x, y, z)
