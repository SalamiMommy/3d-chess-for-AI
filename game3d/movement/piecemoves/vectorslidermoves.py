# game3d/movement/piecemoves/vectorslidermoves.py
"""Exports vector-slider move generator and registers it with the dispatcher."""

from typing import List, TYPE_CHECKING

from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.vectorslidermovement import generate_vector_slider_moves
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

__all__ = ['generate_vector_slider_moves']


@register(PieceType.VECTORSLIDER)
def vectorslider_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_vector_slider_moves(state.cache, state.color, x, y, z)
