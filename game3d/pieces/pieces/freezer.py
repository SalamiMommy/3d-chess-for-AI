"""Freezer – king-like mover + 2-sphere enemy freeze aura."""

from __future__ import annotations
from typing import List, Set, Tuple, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import get_aura_squares, in_bounds
from game3d.common.piece_utils import get_pieces_by_type
from game3d.common.cache_utils import get_occupancy_safe, ensure_int_coords

if TYPE_CHECKING:
    from game3d.pieces.pieces.auras.aura import BoardProto
    from game3d.cache.manager import OptimizedCacheManager

# --------------------------------------------------
#  Public API
# --------------------------------------------------
def generate_freezer_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """King-like single-step generator (re-used)."""
    return generate_king_moves(cache_manager, color, x, y, z)

def get_all_frozen_squares(board: 'BoardProto', controller: Color, cache_manager: 'OptimizedCacheManager | None' = None) -> Set[Tuple[int, int, int]]:
    """
    Get all enemy squares that are frozen by controller's freezers.
    This is called whenever any friendly piece moves.
    """
    if cache_manager is None:
        cache_manager = board.cache_manager

    frozen: Set[Tuple[int, int, int]] = set()
    enemy_color = controller.opposite()

    # Find all friendly freezers using centralized lookup
    freezers = get_pieces_by_type(board, PieceType.FREEZER, controller, cache_manager)
    for coord, _ in freezers:
        # Get all squares in freeze radius
        radius_squares = get_aura_squares(coord)

        # Check each square for enemy pieces using standardized cache access
        for sq in radius_squares:
            target = get_occupancy_safe(cache_manager, sq)
            if target and target.color == enemy_color:
                frozen.add(sq)

    return frozen

# --------------------------------------------------
#  Dispatcher registration
# --------------------------------------------------
@register(PieceType.FREEZER)
def freezer_move_dispatcher(state, x: int, y: int, z: int) -> list[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_freezer_moves(state.cache, state.color, x, y, z)
