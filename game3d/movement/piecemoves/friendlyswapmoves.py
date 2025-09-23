"""Exports Friendly-Swap + king moves (combined) and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.friendlyswapmovement import generate_friendly_swap_moves
from game3d.movement.movetypes.kingmovement import generate_king_moves


def generate_friendly_swap_with_king_moves(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Combine friendly-swap moves + 1-step king moves, deduplicated."""
    swap_moves = generate_friendly_swap_moves(state, x, y, z)
    king_moves = generate_king_moves(state, x, y, z)

    seen = {m.to_coord for m in swap_moves}
    combined = swap_moves[:]
    for m in king_moves:
        if m.to_coord not in seen:
            combined.append(m)
            seen.add(m.to_coord)
    return combined


@register(PieceType.FRIENDLY_SWAPPER)
def friendly_swapper_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Registered dispatcher for Friendly-Swapper moves (now with king moves)."""
    return generate_friendly_swap_with_king_moves(state, x, y, z)


__all__ = ['generate_friendly_swap_with_king_moves']
