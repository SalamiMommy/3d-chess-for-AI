"""Slower – king-like mover + 2-sphere enemy debuff."""

from __future__ import annotations
from typing import List

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import get_aura_squares
from game3d.common.piece_utils import get_pieces_by_type
# --------------------------------------------------
#  Public API
# --------------------------------------------------
def generate_slower_moves(
    cache_manager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """King-like single-step generator (re-used)."""
    return generate_king_moves(cache_manager, color, x, y, z)


def debuffed_squares(board: BoardProto, debuffer_colour: Color, cache_manager) -> Set[Tuple[int, int, int]]:
    debuffed: Set[Tuple[int, int, int]] = set()
    slowers = get_pieces_by_type(board, PieceType.SLOWER, debuffer_colour)
    for coord, _ in slowers:
        for sq in get_aura_squares(coord):
            debuffed.add(sq)
    return debuffed
# --------------------------------------------------
#  Dispatcher registration
# --------------------------------------------------
@register(PieceType.SLOWER)
def slower_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_slower_moves(state.cache, state.color, x, y, z)
