from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.vectorslidermovement import generate_vector_slider_moves

__all__ = ['generate_vector_slider_moves']

@register(PieceType.VECTORSLIDER)
def vectorslider_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_vector_slider_moves(state, x, y, z)
