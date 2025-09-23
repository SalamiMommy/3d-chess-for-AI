"""Exports Face-Cone-Slider move generator and registers it."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.faceconemovement import generate_face_cone_slider_moves


@register(PieceType.FACE_CONE_SLIDER)
def face_cone_slider_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for Face-Cone-Slider moves.
    Delegates to pure geometric cone-slider logic.
    """
    return generate_face_cone_slider_moves(state, x, y, z)


# Re-export for external use
__all__ = ['generate_face_cone_slider_moves']
