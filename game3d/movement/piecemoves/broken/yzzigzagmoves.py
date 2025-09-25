"""Exports YZ-Zig-Zag slider moves (kingless) and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.yzzigzagmovement import generate_yz_zigzag_moves
from game3d.movement.movepiece import Move

@register(PieceType.YZZIGZAG)
def yz_zigzag_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
def yz_zigzag_move_dispatcher    from game3d.game.gamestate import GameState
def yz_zigzag_move_dispatcher    state = GameState(board, color, cache=cache)


__all__ = ['generate_yz_zigzag_moves']
