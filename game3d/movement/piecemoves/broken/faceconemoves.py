"""Exports Face-Cone-Slider move generator and registers it."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.faceconemovement import generate_face_cone_slider_moves
from game3d.movement.movepiece import Move

@register(PieceType.CONESLIDER)
def face_cone_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    """
    Registered dispatcher for Face-Cone-Slider moves.
    Delegates to pure geometric cone-slider logic.
    """
    return generate_face_cone_slider_moves(state, x, y, z)


# Re-export for external use
__all__ = ['generate_face_cone_slider_moves']
