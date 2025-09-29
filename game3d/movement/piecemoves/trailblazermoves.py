"""Exports Trailblazer moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.trailblazermovement import generate_trailblazer_moves
from game3d.movement.movepiece import Move

@register(PieceType.TRAILBLAZER)
def trailblazer_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    # Extract board and color from state
    board = state.board
    color = state.color
    return generate_trailblazer_moves(state.cache, state.color, x, y, z)
__all__ = ['trailblazer_move_dispatcher']
