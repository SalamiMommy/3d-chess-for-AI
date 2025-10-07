"""Geomancy – controller may block unoccupied squares within 3-sphere for 5 plies."""

from __future__ import annotations
from typing import List, Tuple, Optional, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

def block_candidates(board: BoardProto, controller: Color, cache_manager: Optional[OptimizedCacheManager] = None) -> List[Tuple[int, int, int]]:
    """All unoccupied squares within 3-sphere of any friendly GEOMANCER."""
    out: List[Tuple[int, int, int]] = []
    centres = [
        coord for coord, p in board.list_occupied()
        if p.color == controller and p.ptype == PieceType.GEOMANCER
    ]
    for centre in centres:
        for sq in sphere_centre(board, centre, radius=3):
            # Check if square is unoccupied using cache manager if available
            if cache_manager:
                piece = cache_manager.piece_cache.get(sq)
            else:
                # Fallback to board method if cache manager not available
                piece = board.get_piece(sq)

            if piece is None:    # unoccupied only
                out.append(sq)
    return out
