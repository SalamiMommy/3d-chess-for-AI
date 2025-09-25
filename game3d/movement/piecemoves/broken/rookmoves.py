# game3d/movement/piecemoves/rookmoves.py
from typing import List
from game3d.board.board import Board
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.rookmovement import generate_rook_moves
from game3d.movement.movepiece import Move

__all__ = ['generate_rook_moves']

@register(PieceType.ROOK)
def rook_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_rook_moves(state, *coord)
