# game3d/movement/piecemoves/echomoves.py

"""Exports echo move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.echomovement import (
    generate_echo_moves,
    get_bubble_offsets,
    get_anchor_offsets,
    count_valid_echo_moves_from,
    get_echo_theoretical_reach
)

# Re-export core function and helpers for external use
__all__ = [
    'generate_echo_moves',
    'get_bubble_offsets',
    'get_anchor_offsets',
    'count_valid_echo_moves_from',
    'get_echo_theoretical_reach'
]


@register(PieceType.ECHO)
def echo_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for echo moves.
    Simply delegates to the core move generator.
    """
    return generate_echo_moves(state, x, y, z)
