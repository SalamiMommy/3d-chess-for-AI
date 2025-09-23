# game3d/movement/piecemoves/knightmoves.py

"""Exports knight move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.knightmovement import generate_knight_moves

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_knight_moves']


@register(PieceType.KNIGHT)
def knight_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for knight moves.
    Simply delegates to the core move generator.
    """
    return generate_knight_moves(state, x, y, z)
