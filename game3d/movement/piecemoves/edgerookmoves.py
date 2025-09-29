# game3d/movement/piecemoves/edgerookmoves.py
"""Exports edge-rook move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.edgerookmovement import generate_edgerook_moves
from game3d.movement.movepiece import Move

# Re-export core function for external use
__all__ = ['generate_edgerook_moves']


@register(PieceType.EDGEROOK)
def edgerook_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_edgerook_moves(state.cache, state.color, x, y, z)
