"""Exports Spiral moves and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.spiralmovement import generate_spiral_moves


@register(PieceType.SPIRAL)
def spiral_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Registered dispatcher for Counter-Clockwise Spiral moves."""
    return generate_spiral_moves(state, x, y, z)


__all__ = ['generate_spiral_moves']
