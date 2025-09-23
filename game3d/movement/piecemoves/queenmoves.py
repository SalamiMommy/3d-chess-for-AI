# game3d/movement/piecemoves/queenmoves.py

"""Exports queen move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.queenmovement import generate_queen_moves

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_queen_moves']


@register(PieceType.QUEEN)
def queen_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for queen moves.
    Simply delegates to the core move generator.
    """
    return generate_queen_moves(state, x, y, z)
