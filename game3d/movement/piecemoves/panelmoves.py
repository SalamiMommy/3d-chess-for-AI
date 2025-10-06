# game3d/movement/piecemoves/panelmoves.py
# game3d/movement/piecemoves/panelmoves.py
"""Exports panel move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

from game3d.movement.registry import register
from game3d.movement.movetypes.panelmovement import generate_panel_moves

from game3d.movement.movepiece import Move


@register(PieceType.PANEL)
def panel_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_panel_moves(state.cache, state.color, x, y, z)


# Re-export core function and helpers for external use
__all__ = [
    'generate_panel_moves',
    'get_panel_offsets',
    'count_valid_panel_moves_from',
    'get_panel_theoretical_reach'
]
