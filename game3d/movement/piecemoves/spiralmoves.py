"""Exports Spiral moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.spiralmovement import generate_spiral_moves
from game3d.movement.movepiece import Move

@register(PieceType.SPIRAL)
def spiral_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_spiral_moves(state, *coord)


__all__ = ['generate_spiral_moves']
