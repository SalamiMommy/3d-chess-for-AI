"""Exports YZ-Zig-Zag slider moves (kingless) and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.yzzigzagmovement import generate_yz_zigzag_moves


@register(PieceType.YZ_ZIGZAG_SLIDER)
def yz_zigzag_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Registered dispatcher for YZ-Zig-Zag slider."""
    return generate_yz_zigzag_moves(state, x, y, z)


__all__ = ['generate_yz_zigzag_moves']
