# game3d/movement/piecemoves/trigonalbishopmoves.py

"""Exports trigonal bishop move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.trigonalbishopmovement import generate_trigonal_bishop_moves

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_trigonal_bishop_moves']


@register(PieceType.TRIGONALBISHOP)
def trigonalbishop_dispatcher(state: GameState, x: int, y: int, z: int) -> List['Move']:
    """
    Registered dispatcher for trigonal bishop moves.
    Simply delegates to the core move generator.
    """
    return generate_trigonal_bishop_moves(state, x, y, z)
