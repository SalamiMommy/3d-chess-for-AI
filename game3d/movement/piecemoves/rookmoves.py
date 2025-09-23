# game3d/movement/piecemoves/rookmoves.py

"""Exports rook move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.rookmovement import generate_rook_moves

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_rook_moves']


@register(PieceType.ROOK)
def rook_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for rook moves.
    Simply delegates to the core move generator.
    """
    return generate_rook_moves(state, x, y, z)
