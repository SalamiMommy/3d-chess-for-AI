"""Exports Reflecting-Bishop moves and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.reflectingbishopmovement import generate_reflecting_bishop_moves


@register(PieceType.REFLECTING_BISHOP)
def reflecting_bishop_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Registered dispatcher for Reflecting-Bishop moves."""
    return generate_reflecting_bishop_moves(state, x, y, z)


__all__ = ['generate_reflecting_bishop_moves']
