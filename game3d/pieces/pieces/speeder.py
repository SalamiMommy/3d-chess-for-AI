"""Speeder – king-like mover + 2-sphere friendly buff."""

from __future__ import annotations
from typing import List

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import get_aura_squares
# --------------------------------------------------
#  Public API
# --------------------------------------------------
def generate_speeder_moves(
    cache_manager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """King-like single-step generator (re-used)."""
    return generate_king_moves(cache_manager, color, x, y, z)


def buffed_squares(board: BoardProto, buffer_colour: Color, cache_manager) -> Set[Tuple[int, int, int]]:
    """Return friendly squares within 2-sphere of any friendly SPEEDER."""
    buffed: Set[Tuple[int, int, int]] = set()
    speeders = get_pieces_by_type(board, PieceType.SPEEDER, buffer_colour)
    for coord, _ in speeders:
        for sq in get_aura_squares(coord):
            if not in_bounds(sq):  # Already checked in get_aura_squares, but kept for clarity
                continue
            # Use occupancy cache consistently
            target = cache_manager.occupancy.get(sq) if cache_manager else None
            if target is not None and target.color == buffer_colour:
                buffed.add(sq)
    return buffed
# --------------------------------------------------
#  Dispatcher registration
# --------------------------------------------------
@register(PieceType.SPEEDER)
def speeder_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_speeder_moves(state.cache, state.color, x, y, z)
