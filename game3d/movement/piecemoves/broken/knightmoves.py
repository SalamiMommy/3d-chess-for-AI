# game3d/movement/piecemoves/knightmoves.py

"""Exports knight move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.knightmovement import generate_knight_moves
from game3d.movement.movepiece import Move
# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_knight_moves']


@register(PieceType.KNIGHT)
def knight_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    state = GameState(board, color, cache=cache)
    return generate_knight_moves(state, r, c, z)
