"""White-Hole – moves like a Speeder and pushes enemies 1 step away at turn end."""

from __future__ import annotations
from typing import List, Dict, Tuple, TYPE_CHECKING, Set  # ADDED: Set import

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import add_coords, in_bounds, chebyshev_distance
from game3d.common.piece_utils import get_pieces_by_type
from game3d.common.cache_utils import is_occupied_safe, ensure_int_coords  # CHANGED: is_occupied_safe

if TYPE_CHECKING:
    from game3d.board.board import BoardProto
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def _away(pos: Tuple[int, int, int], target: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """1 Chebyshev step from pos *away* from target."""
    x, y, z = pos
    tx, ty, tz = target
    dx = 0 if x == tx else (1 if tx < x else -1)
    dy = 0 if y == ty else (1 if ty < y else -1)
    dz = 0 if z == tz else (1 if tz < z else -1)
    return add_coords(pos, (dx, dy, dz))

def generate_whitehole_moves(
    cache_manager: 'OptimizedCacheManager',  # STANDARDIZED
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """White-Hole moves exactly like a Speeder (king single steps)."""
    return generate_king_moves(cache_manager, color, x, y, z)  # STANDARDIZED: pass cache_manager

def push_candidates(
    cache_manager: 'OptimizedCacheManager',  # STANDARDIZED: Single parameter
    controller: Color,
) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Return dict {enemy_square: push_target} for every enemy within
    2-sphere of any friendly WHITE_HOLE.
    """
    out: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

    holes = [
        coord for coord, piece in cache_manager.get_pieces_of_color(controller)
        if piece.ptype == PieceType.WHITEHOLE
    ]

    if not holes:
        return out

    enemy_color = controller.opposite()

    # Use cache manager's piece iteration
    for coord, piece in cache_manager.get_pieces_of_color(enemy_color):
        for hole in holes:
            if chebyshev_distance(coord, hole) <= 2:
                push = _away(coord, hole)
                # Use standardized cache utils
                if in_bounds(push) and not is_occupied_safe(cache_manager, push):
                    out[coord] = push
                break
    return out

@register(PieceType.WHITEHOLE)
def whitehole_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    # STANDARDIZED: Use cache_manager property
    return generate_whitehole_moves(state.cache_manager, state.color, x, y, z)
