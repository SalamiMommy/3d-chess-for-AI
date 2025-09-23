"""Exports XZ-Zig-Zag slider moves (kingless) and registers them."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.xzzigzagmovement import generate_xz_zigzag_moves


@register(PieceType.XZ_ZIGZAG_SLIDER)
def xz_zigzag_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """Registered dispatcher for XZ-Zig-Zag slider."""
    return generate_xz_zigzag_moves(state, x, y, z)


__all__ = ['generate_xz_zigzag_moves']
