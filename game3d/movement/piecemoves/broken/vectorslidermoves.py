from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.vectorslidermovement import generate_vector_slider_moves
from game3d.movement.movepiece import Move
__all__ = ['generate_vector_slider_moves']

@register(PieceType.VECTORSLIDER)
def vectorslider_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
def vectorslider_move_dispatcher    from game3d.game.gamestate import GameState
def vectorslider_move_dispatcher    state = GameState(board, color, cache=cache)
