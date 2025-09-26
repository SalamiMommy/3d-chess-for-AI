#game3d/movement/piecemoves/swappermoves.py
"""Exports swapper move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movetypes.swapmovement import generate_swap_moves
from game3d.movement.movepiece import Move

# Re-export the move generators for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_swapper_moves']


@register(PieceType.SWAPPER)
def swapper_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_swapper_moves(state, x, y, z)


def generate_swapper_moves(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    """Combines king moves and swap moves for the Swapper piece."""
    moves: List[Move] = []
    moves.extend(generate_king_moves(state, x, y, z))
    moves.extend(generate_swap_moves(state, x, y, z))
    return moves
