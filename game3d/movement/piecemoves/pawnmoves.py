# game3d/movement/piecemoves/pawnmoves.py

"""Exports pawn move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.pawnmovement import generate_pawn_moves
from game3d.movement.movepiece import Move

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_pawn_moves']


@register(PieceType.PAWN)
def pawn_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_pawn_moves(state, *coord)
