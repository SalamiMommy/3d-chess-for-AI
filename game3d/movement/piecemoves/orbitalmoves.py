# game3d/movement/piecemoves/orbitalmoves.py

"""Exports orbital move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # ← run-time no-op
    from game3d.game.gamestate import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.orbitalmovement import generate_orbital_moves
from game3d.movement.movepiece import Move
# Re-export core function and helpers for external use
__all__ = [
    'generate_orbital_moves',
    'get_orbital_offsets',
    'count_valid_orbital_moves_from'
]


@register(PieceType.ORBITER)
def orbital_move_dispatcher(board, color, *coord, cache=None) -> List[Move]:
    from game3d.game.gamestate import GameState
    state = GameState(board, color, cache=cache)
    return generate_orbital_moves(state, *coord)
