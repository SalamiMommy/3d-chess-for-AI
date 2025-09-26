# game3d/movement/piecemoves/priestmoves.py
"""Exports priest move generator (same as king) and registers it with the dispatcher."""

from typing import List, TYPE_CHECKING

from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# Alias for clarity — generate_priest_moves is the same function as generate_king_moves
generate_priest_moves = generate_king_moves

__all__ = ['generate_priest_moves']


@register(PieceType.PRIEST)
def priest_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_priest_moves(state, x, y, z)
