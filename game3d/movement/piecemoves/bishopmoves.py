# game3d/movement/piecemoves/bishopmoves.py

"""Exports bishop move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.bishopmovement import generate_bishop_moves

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_bishop_moves']


@register(PieceType.BISHOP)
def bishop_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for bishop moves.
    Simply delegates to the core move generator.
    """
    return generate_bishop_moves(state, x, y, z)
