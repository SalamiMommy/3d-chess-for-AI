# game3d/movement/piecemoves/nebulamoves.py
"""Exports nebula move generator and registers it with the dispatcher."""

from typing import List
from game3d.pieces.enums import PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.nebulamovement import (
    generate_nebula_moves,
    get_nebula_offsets,
    count_valid_nebula_moves_from,
    get_nebula_reach_volume
)
from game3d.movement.movepiece import Move

# Re-export core function and helpers for external use
__all__ = [
    'generate_nebula_moves',
    'get_nebula_offsets',
    'count_valid_nebula_moves_from',
    'get_nebula_reach_volume'
]


@register(PieceType.NEBULA)
def nebula_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_nebula_moves(state, x, y, z)
