"""Exports Friendly-Swap + king moves (combined) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.friendlyswapmovement import generate_friendly_swap_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

def generate_friendly_swap_with_king_moves(board, color, *coord, cache=None) -> List[Move]:
def generate_friendly_swap_with_king_moves    from game3d.game.gamestate import GameState
def generate_friendly_swap_with_king_moves    state = GameState(board, color, cache=cache)
    king_moves = generate_king_moves(state, x, y, z)

    seen = {m.to_coord for m in swap_moves}
    combined = swap_moves[:]
    for m in king_moves:
        if m.to_coord not in seen:
            combined.append(m)
            seen.add(m.to_coord)
    return combined


@register(PieceType.SWAPPER)
def friendly_swap_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    """Registered dispatcher for Friendly-Swapper moves (now with king moves)."""
    return generate_friendly_swap_with_king_moves(state, x, y, z)


__all__ = ['generate_friendly_swap_with_king_moves']
