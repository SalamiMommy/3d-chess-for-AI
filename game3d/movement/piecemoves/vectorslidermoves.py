# game3d/movement/piecemoves/vectorslidermoves.py

"""Exports vector slider move generator and registers it for multiple slider types."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.vectorslidermovement import generate_vector_slider_moves

# Re-export for external use
__all__ = ['generate_vector_slider_moves']


# Register for all slider types
@register(PieceType.VECTOR_SLIDER)
@register(PieceType.SPACE_BISHOP)
@register(PieceType.KNIGHT_SLIDER)
@register(PieceType.CUSTOM_SLIDER_1)
def vector_slider_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for all vector slider types.
    Delegates to core vector slider move generator.
    """
    return generate_vector_slider_moves(state, x, y, z)
