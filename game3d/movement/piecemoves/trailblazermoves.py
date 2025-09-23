"""Exports Trailblazer moves and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.trailblazermovement import generate_trailblazer_moves


@register(PieceType.TRAILBLAZER)
def trailblazer_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Registered dispatcher for Trailblazer moves."""
    return generate_trailblazer_moves(state, x, y, z)


__all__ = ['generate_trailblazer_moves']
