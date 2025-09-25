# game3d/movement/piecemoves/priestmoves.py

"""Exports priest move generator (same as king) and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
# Re-export the move generator — it’s identical to king’s
__all__ = ['generate_priest_moves']


# Alias for clarity — generate_priest_moves is the same function as generate_king_moves
generate_priest_moves = generate_king_moves


@register(PieceType.PRIEST)
def priest_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    """
    Registered dispatcher for priest moves.
    Delegates to king movement logic — priest moves exactly like a king.
    """
    return generate_priest_moves(state, x, y, z)
