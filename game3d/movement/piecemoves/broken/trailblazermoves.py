"""Exports Trailblazer moves and registers them."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.trailblazermovement import generate_trailblazer_moves
from game3d.movement.movepiece import Move

@register(PieceType.TRAILBLAZER)
def trailblazer_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
def trailblazer_move_dispatcher    from game3d.game.gamestate import GameState
def trailblazer_move_dispatcher    state = GameState(board, color, cache=cache)


__all__ = ['generate_trailblazer_moves']
