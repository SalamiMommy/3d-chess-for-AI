# game3d/movement/piecemoves/queenmoves.py

"""Exports queen move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.queenmovement import generate_queen_moves
from game3d.movement.movepiece import Move
# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_queen_moves']


@register(PieceType.QUEEN)
def queen_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_queen_moves(state, *coord)
