"""Exports YZ-Zig-Zag slider moves (kingless) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.yzzigzagmovement import generate_yz_zigzag_moves
from game3d.movement.movepiece import Move

@register(PieceType.YZZIGZAG)
def yz_zigzag_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    """Registered dispatcher for YZ-Zig-Zag slider."""
    return generate_yz_zigzag_moves(state, *coord)


__all__ = ['generate_yz_zigzag_moves']
