# game3d/movement/piecemoves/pawnmoves.py
"""Exports pawn move generator and registers it with the dispatcher."""

from typing import List, TYPE_CHECKING

from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.pawnmovement import generate_pawn_moves
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_pawn_moves']


@register(PieceType.PAWN)
def pawn_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_pawn_moves(state.cache, state.color, x, y, z)
