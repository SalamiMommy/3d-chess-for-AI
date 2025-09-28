"""Exports knight move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.knightmovement import generate_knight_moves
from game3d.movement.movepiece import Move

@register(PieceType.KNIGHT)
def knight_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    # ✅ Pass cache as second argument
    return generate_knight_moves(state.board, state.cache, state.color, x, y, z)

# Re-export for external use
__all__ = ['generate_knight_moves']
