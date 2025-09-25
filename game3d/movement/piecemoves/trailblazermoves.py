"""Exports Trailblazer moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.trailblazermovement import generate_trailblazer_moves
from game3d.movement.movepiece import Move

@register(PieceType.TRAILBLAZER)
def trailblazer_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_trailblazer_moves(state, *coord)


__all__ = ['generate_trailblazer_moves']
