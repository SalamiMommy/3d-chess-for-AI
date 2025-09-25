# game3d/movement/piecemoves/echomoves.py

"""Exports echo move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.echomovement import (
    generate_echo_moves,
    get_bubble_offsets,
    get_anchor_offsets,
    count_valid_echo_moves_from,
    get_echo_theoretical_reach
)
from game3d.movement.movepiece import Move
# Re-export core function and helpers for external use
__all__ = [
    'generate_echo_moves',
    'get_bubble_offsets',
    'get_anchor_offsets',
    'count_valid_echo_moves_from',
    'get_echo_theoretical_reach'
]


@register(PieceType.ECHO)
def echo_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_echo_moves(state, *coord)
