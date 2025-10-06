# game3d/movement/piecemoves/blackholemoves.py
"""Exports blackhole move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

# Re-export the move generator for use by other modules (e.g., attacks, UI, AI)
__all__ = ['generate_blackhole_moves']


@register(PieceType.BLACKHOLE)
def blackhole_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_king_moves(state.cache, state.color, x, y, z)


def generate_blackhole_moves(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    """Alias for king moves since BlackHole uses the same movement pattern."""
    return generate_king_moves(state.cache, state.color, x, y, z)
