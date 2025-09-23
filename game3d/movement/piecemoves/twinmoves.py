# game3d/movement/piecemoves/twinmoves.py

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves

__all__ = ['generate_twin_moves']

generate_twin_moves = generate_king_moves  # Identical movement


@register(PieceType.TWIN)
def twin_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_twin_moves(state, x, y, z)
