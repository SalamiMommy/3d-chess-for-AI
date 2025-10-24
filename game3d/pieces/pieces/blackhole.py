# blackhole.py - FIXED
"""Black-Hole – moves like a Speeder and drags enemies 1 step closer at turn end."""

from __future__ import annotations
from typing import List, Dict, Tuple, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import add_coords, in_bounds, chebyshev_distance
from game3d.common.cache_utils import ensure_int_coords

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

def _toward(pos: Tuple[int, int, int], target: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """1 Chebyshev step from pos *toward* target."""
    x, y, z = pos
    tx, ty, tz = target
    dx = 0 if x == tx else (1 if tx > x else -1)
    dy = 0 if y == ty else (1 if ty > y else -1)
    dz = 0 if z == tz else (1 if tz > z else -1)
    return add_coords(pos, (dx, dy, dz))

def generate_blackhole_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Black-Hole moves exactly like a Speeder (king single steps)."""
    return generate_king_moves(cache_manager, color, x, y, z)

def suck_candidates(
    cache_manager: 'OptimizedCacheManager',  # STANDARDIZED: Single parameter
    controller: Color,
) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Return dict {enemy_square: pull_target} for every enemy within
    2-sphere of any friendly BLACK_HOLE.
    """
    out: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

    # FIXED: Use cache_manager to get pieces
    holes = [
        coord for coord, piece in cache_manager.get_pieces_of_color(controller)
        if piece.ptype == PieceType.BLACKHOLE
    ]

    if not holes:
        return out

    enemy_color = controller.opposite()

    # Use cache manager's piece iteration
    for coord, piece in cache_manager.get_pieces_of_color(enemy_color):
        for hole in holes:
            if chebyshev_distance(coord, hole) <= 2:
                pull = _toward(coord, hole)
                # FIXED: Use cache_manager.get_piece() for occupancy check
                if in_bounds(pull) and cache_manager.get_piece(pull) is None:
                    out[coord] = pull
                break  # pull toward first hole only
    return out

@register(PieceType.BLACKHOLE)
def blackhole_move_dispatcher(state: 'GameState', x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    # STANDARDIZED: Use cache_manager property
    return generate_blackhole_moves(state.cache_manager, state.color, x, y, z)

__all__ = ["generate_blackhole_moves", "suck_candidates"]
