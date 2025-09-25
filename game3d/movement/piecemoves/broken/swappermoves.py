"""Exports king move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movetypes.swapmovement import generate_swap_moves
from game3d.movement.movepiece import Move

# Re-export the move generators for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_king_moves', 'generate_swap_moves']


@register(PieceType.SWAPPER)
def swapper_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
def swapper_move_dispatcher    from game3d.game.gamestate import GameState
def swapper_move_dispatcher    state = GameState(board, color, cache=cache)
    Combines standard king-style moves with friendly-piece swaps.
    """
    moves: List[Move] = []
    moves.extend(generate_king_moves(state, x, y, z))
    moves.extend(generate_swap_moves(state, x, y, z))
    return moves
