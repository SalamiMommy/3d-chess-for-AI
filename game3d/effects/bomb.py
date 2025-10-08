"""Bomb – detonates on capture OR on self-move to same square; kills enemies in 2-sphere (no kings)."""

from __future__ import annotations
from typing import List, Tuple, Dict
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds

def detonate(board: BoardProto, trigger_sq: Tuple[int, int, int], current_color: Color) -> List[Tuple[int, int, int]]:
    """
    Simplified detonation - use cache manager directly.
    """
    cleared: List[Tuple[int, int, int]] = []

    # Use cache manager if available
    cache_manager = getattr(board, 'cache_manager', None)

    # Get sphere surface
    from game3d.effects.auras.aura import sphere_centre
    for sq in sphere_centre(board, trigger_sq, radius=2):
        # Fast piece lookup using cache
        if cache_manager:
            victim = cache_manager.piece_cache.get(sq)
        else:
            victim = board.get_piece(sq)

        if victim is None or victim.color == current_color:
            continue
        if victim.ptype == PieceType.KING:
            continue

        cleared.append(sq)

    return cleared
