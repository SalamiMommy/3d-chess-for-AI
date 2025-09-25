"""Exports (3,2) knight moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.knight32movement import generate_knight32_moves
from game3d.movement.movepiece import Move

@register(PieceType.KNIGHT32)
def knight32_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_knight32_moves(state, *coord)


__all__ = ['generate_knight32_moves']
