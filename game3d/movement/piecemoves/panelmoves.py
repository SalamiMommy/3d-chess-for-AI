# game3d/movement/piecemoves/panelmoves.py

"""Exports panel move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.panelmovement import (
    generate_panel_moves,
    get_panel_offsets,
    count_valid_panel_moves_from,
    get_panel_theoretical_reach
)

# Re-export core function and helpers for external use
__all__ = [
    'generate_panel_moves',
    'get_panel_offsets',
    'count_valid_panel_moves_from',
    'get_panel_theoretical_reach'
]


@register(PieceType.PANEL)
def panel_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for panel moves.
    Simply delegates to the core move generator.
    """
    return generate_panel_moves(state, x, y, z)
