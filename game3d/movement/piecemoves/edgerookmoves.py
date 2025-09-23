# game3d/movement/piecemoves/edgerookmoves.py

"""Exports edge-rook move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.edgerookmovement import generate_edge_rook_moves

# Re-export core function for external use
__all__ = ['generate_edge_rook_moves']


@register(PieceType.EDGEROOK)
def edge_rook_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for edge-rook moves.
    Simply delegates to the core move generator.
    """
    return generate_edge_rook_moves(state, x, y, z)
