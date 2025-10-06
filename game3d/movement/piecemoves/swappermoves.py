#game3d/movement/piecemoves/swappermoves.py
"""Exports swapper move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movetypes.swapmovement import generate_swapper_moves  # ← now matches
from game3d.movement.movepiece import Move

@register(PieceType.SWAPPER)
def swapper_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    """Combines king moves and swap moves for the Swapper piece."""
    king_moves = generate_king_moves(state.cache, state.color, x, y, z)
    swap_moves = generate_swapper_moves(state.cache, state.color, x, y, z)
    return king_moves + swap_moves

# Re-export for external use
__all__ = ['generate_swapper_moves']
