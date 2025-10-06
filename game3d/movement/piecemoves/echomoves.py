# game3d/movement/piecemoves/echomoves.py
"""Echo-move dispatcher – thin wrapper around echomovement."""

from typing import TYPE_CHECKING, List

from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.echomovement import generate_echo_moves
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState


@register(PieceType.ECHO)
def echo_move_dispatcher(state: "GameState", x: int, y: int, z: int) -> List[Move]:
    """Generate every legal Echo move for the piece at (x, y, z)."""
    return generate_echo_moves(state.cache, state.color, x, y, z)
