"""Freezer – king-like mover + 2-sphere enemy freeze aura."""

from __future__ import annotations
from typing import List

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.common import get_aura_squares

# --------------------------------------------------
#  Public API
# --------------------------------------------------
def generate_freezer_moves(
    cache_manager,
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """King-like single-step generator (re-used)."""
    return generate_king_moves(cache_manager, color, x, y, z)


def get_freeze_radius(freezer_pos: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """Get all squares within 2-sphere of a freezer position."""
    return get_aura_squares(freezer_pos)  # Use common helper

def get_all_frozen_squares(board, controller: Color) -> Set[Tuple[int, int, int]]:
    """
    Get all enemy squares that are frozen by controller's freezers.
    This is called whenever any friendly piece moves.
    """
    frozen: Set[Tuple[int, int, int]] = set()
    enemy_color = controller.opposite()
    cache_manager = board.cache_manager if hasattr(board, 'cache_manager') else None

    # Find all friendly freezers using centralized lookup
    freezers = get_pieces_by_type(board, PieceType.FREEZER, controller)
    for coord, _ in freezers:
        # Get all squares in freeze radius
        radius_squares = get_freeze_radius(coord)

        # Check each square for enemy pieces using occupancy cache
        for sq in radius_squares:
            target = cache_manager.occupancy.get(sq) if cache_manager else None
            if target and target.color == enemy_color:
                frozen.add(sq)

    return frozen
# --------------------------------------------------
#  Dispatcher registration
# --------------------------------------------------
@register(PieceType.FREEZER)
def freezer_move_dispatcher(state, x: int, y: int, z: int) -> list[Move]:
    return generate_freezer_moves(state.cache, state.color, x, y, z)
