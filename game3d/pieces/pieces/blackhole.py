"""Black-Hole – moves like a Speeder and drags enemies 1 step closer at turn end."""

from __future__ import annotations
from typing import List, Dict, Tuple, TYPE_CHECKING

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import add_coords, in_bounds, chebyshev_distance
from game3d.common.piece_utils import get_pieces_by_type
from game3d.common.cache_utils import get_occupancy_safe, ensure_int_coords

if TYPE_CHECKING:
    from game3d.pieces.pieces.auras.aura import BoardProto
    from game3d.cache.manager import OptimizedCacheManager

# ------------------------------------------------------------------
#  Internal helpers
# ------------------------------------------------------------------
def _toward(pos: Tuple[int, int, int], target: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """1 Chebyshev step from pos *toward* target."""
    x, y, z = pos
    tx, ty, tz = target
    dx = 0 if x == tx else (1 if tx > x else -1)
    dy = 0 if y == ty else (1 if ty > y else -1)
    dz = 0 if z == tz else (1 if tz > z else -1)
    return add_coords(pos, (dx, dy, dz))

def _pieces_from_cache(cache_manager: 'OptimizedCacheManager', color: Color):
    """Fast-path helper: produce (coord, Piece) tuples."""
    for coord, piece in cache_manager.get_pieces_of_color(color):
        yield coord, piece

# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------
def generate_blackhole_moves(
    cache_manager: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    """Black-Hole moves exactly like a Speeder (king single steps)."""
    return generate_king_moves(cache_manager, color, x, y, z)

def suck_candidates(
    board: 'BoardProto',
    controller: Color,
    cache_manager: 'OptimizedCacheManager | None' = None
) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Return dict {enemy_square: pull_target} for every enemy within
    2-sphere of any friendly BLACK_HOLE.  Pull target is 1 step toward
    the nearest hole (first hole found).
    """
    out: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

    if cache_manager is None:
        cache_manager = board.cache_manager

    holes: list[Tuple[int, int, int]] = [
        coord for coord, _ in get_pieces_by_type(board, PieceType.BLACKHOLE, controller, cache_manager)
    ]
    if not holes:
        return out

    enemy_color = controller.opposite()

    # Use consistent cache-based piece iteration
    for coord, piece in _pieces_from_cache(cache_manager, enemy_color):
        for hole in holes:
            if chebyshev_distance(coord, hole) <= 2:
                pull = _toward(coord, hole)
                if in_bounds(pull) and not is_occupied_safe(cache_manager, pull):
                    out[coord] = pull
                break  # pull toward first hole only
    return out

# ------------------------------------------------------------------
#  Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.BLACKHOLE)
def blackhole_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_blackhole_moves(state.cache, state.color, x, y, z)
