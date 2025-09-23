# game3d/movement/piecemoves/orbitalmoves.py

"""Exports orbital move generator and registers it with the dispatcher."""

from typing import List
from pieces.enums import PieceType
from game.state import GameState
from game3d.movement.registry import register
from game3d.movement.movetypes.orbitalmovement import generate_orbital_moves

# Re-export core function and helpers for external use
__all__ = [
    'generate_orbital_moves',
    'get_orbital_offsets',
    'count_valid_orbital_moves_from'
]


@register(PieceType.ORBITAL)
def orbital_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    """
    Registered dispatcher for orbital moves.
    Simply delegates to the core move generator.
    """
    return generate_orbital_moves(state, x, y, z)
