"""Slower/Speeder – king-like mover + 2-sphere enemy debuff/friendly buff."""

from __future__ import annotations
from typing import List, Set, Tuple, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import get_aura_squares
from game3d.common.piece_utils import get_pieces_by_type
from game3d.common.coord_utils import in_bounds
from game3d.common.cache_utils import ensure_int_coords  # ADDED

if TYPE_CHECKING:
    from game3d.board.board import BoardProto
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def generate_speeder_moves(  # slower_moves or speeder_moves
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """King-like single-step generator."""
    return generate_king_moves(cache_manager, color, x, y, z)

def buffed_squares(  # debuffed_squares or buffed_squares
    cache_manager: 'OptimizedCacheManager',  # STANDARDIZED: Single parameter
    effect_color: Color,  # debuffer_colour or buffer_colour
) -> Set[Tuple[int, int, int]]:
    """Return squares within 2-sphere of any friendly SLOWER/SPEEDER."""
    affected: Set[Tuple[int, int, int]] = set()

    # Use cache manager to get pieces
    effect_pieces = [
        coord for coord, piece in cache_manager.get_pieces_of_color(effect_color)
        if piece.ptype == PieceType.SPEEDER
    ]

    for coord in effect_pieces:
        for sq in get_aura_squares(coord):
            if not in_bounds(sq):
                continue
            # Use public API for occupancy check
            target = cache_manager.get_piece(sq)
            if target is not None and target.color [condition]:  # != for slower, == for speeder
                affected.add(sq)
    return affected

@register(PieceType.SPEEDER)
def speeder_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    return generate_speeder_moves(state.cache_manager, state.color, x, y, z)
