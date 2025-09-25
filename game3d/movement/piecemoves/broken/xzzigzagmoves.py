"""Exports XZ-Zig-Zag slider moves (kingless) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.xzzigzagmovement import generate_xz_zigzag_moves
from game3d.movement.movepiece import Move

@register(PieceType.XZZIGZAG)
def xz_zigzag_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
def xz_zigzag_move_dispatcher    from game3d.game.gamestate import GameState
def xz_zigzag_move_dispatcher    state = GameState(board, color, cache=cache)


__all__ = ['generate_xz_zigzag_moves']
