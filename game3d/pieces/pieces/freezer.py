# freezer.py - FIXED
"""Freezer – king-like mover + 2-sphere enemy freeze aura."""

from __future__ import annotations
from typing import List, Set, Tuple, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import get_aura_squares, in_bounds
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def generate_freezer_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """King-like single-step generator."""
    return generate_king_moves(cache_manager, color, x, y, z)

def get_all_frozen_squares(
    cache_manager: 'OptimizedCacheManager',  # STANDARDIZED: Single parameter
    controller: Color,
) -> Set[Tuple[int, int, int]]:
    """
    Get all enemy squares that are frozen by controller's freezers.
    """
    frozen: Set[Tuple[int, int, int]] = set()
    enemy_color = controller.opposite()

    # FIXED: Use cache_manager to get pieces
    freezers = [
        coord for coord, piece in cache_manager.get_pieces_of_color(controller)
        if piece.ptype == PieceType.FREEZER
    ]

    for coord in freezers:
        # Get all squares in freeze radius
        radius_squares = get_aura_squares(coord)

        # Check each square for enemy pieces using cache_manager
        for sq in radius_squares:
            target = cache_manager.get_piece(sq)
            if target and target.color == enemy_color:
                frozen.add(sq)

    return frozen

@register(PieceType.FREEZER)
def freezer_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    # STANDARDIZED: Use cache_manager property
    return generate_freezer_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_freezer_moves", "get_all_frozen_squares"]
