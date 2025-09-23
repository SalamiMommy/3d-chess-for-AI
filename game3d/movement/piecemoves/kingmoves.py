# game3d/movement/piecemoves/kingmoves.py

"""Exports king move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_king_moves']


@register(PieceType.KING)
def king_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for king moves.
    Simply delegates to the core move generator.
    """
    return generate_king_moves(state, x, y, z)
