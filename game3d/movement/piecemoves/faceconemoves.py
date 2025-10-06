#game3d/movement/piecemoves/faceconemoves.py
"""Exports Face-Cone-Slider move generator and registers it."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.faceconemovement import generate_face_cone_slider_moves
from game3d.movement.movepiece import Move

@register(PieceType.CONESLIDER)
def face_cone_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_face_cone_slider_moves(state.cache, state.color, x, y, z)

# Re-export for external use
__all__ = ['generate_face_cone_slider_moves']
