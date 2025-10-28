# effect_utils.py
from __future__ import annotations
from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.movement.movepiece import Move
    from game3d.board.board import Board
    from game3d.common.enums import Color
else:
    # Import at runtime
    from game3d.common.enums import Color

from game3d.common.piece_utils import get_piece_effect_type

def apply_standard_effects(
    cache_manager: "OptimizedCacheManager",
    move: "Move",
    mover: Color,
    current_ply: int,
    board: "Board"
) -> Set[str]:
    """Apply standard effects in consistent order."""
    affected_caches = set()

    # Determine affected caches
    from_piece = cache_manager.occupancy_cache.get(move.from_coord)
    if from_piece:
        effect_type = get_piece_effect_type(from_piece.ptype)
        if effect_type:
            affected_caches.add(effect_type)

    # Apply effects in consistent order using pre-defined tuple
    effect_order = ('aura', 'trailblaze', 'geomancy', 'attacks')
    for effect_name in effect_order:
        if effect_name in affected_caches:
            cache = cache_manager._get_cache_by_name(effect_name)
            if cache is not None and hasattr(cache, 'apply_move'):
                cache.apply_move(move, mover, current_ply, board)

    return affected_caches
