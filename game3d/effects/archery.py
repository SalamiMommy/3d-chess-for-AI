"""Archery – controller may attack (capture) any enemy within 2-sphere without moving."""

from __future__ import annotations
from typing import List, Tuple, Set, Optional, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager

def archery_targets(board: BoardProto, controller: Color, cache_manager: Optional[OptimizedCacheManager] = None) -> List[Tuple[int, int, int]]:
    """All enemy pieces within 2-sphere of any friendly ARCHER."""
    out: List[Tuple[int, int, int]] = []
    centres = [
        coord for coord, p in board.list_occupied()
        if p.color == controller and p.ptype == PieceType.ARCHER
    ]
    seen: Set[Tuple[int, int, int]] = set()
    for centre in centres:
        for sq in sphere_centre(board, centre, radius=2):
            if sq in seen:
                continue
            # Use cache manager to get piece at square
            if cache_manager:
                victim = cache_manager.piece_cache.get(sq)
            else:
                # Fallback to board method if cache manager not available
                victim = board.cache_manager.occupancy.get(sq) if cache_manager else board.get_piece(sq)

            if victim is not None and victim.color != controller:
                out.append(sq)
                seen.add(sq)
    return out


def has_line_of_sight(archer_pos: Tuple[int, int, int], target_pos: Tuple[int, int, int],
                     board: BoardProto, cache_manager: Optional[OptimizedCacheManager] = None) -> bool:
    """
    Check if there's a clear line of sight between archer and target.
    This is a simplified version that just checks if the path is clear.
    """
    # Calculate direction vector
    dx = target_pos[0] - archer_pos[0]
    dy = target_pos[1] - archer_pos[1]
    dz = target_pos[2] - archer_pos[2]

    # Normalize to unit steps
    steps = max(abs(dx), abs(dy), abs(dz))
    if steps == 0:
        return False

    step_x = dx // steps if dx != 0 else 0
    step_y = dy // steps if dy != 0 else 0
    step_z = dz // steps if dz != 0 else 0

    # Check each square along the path (excluding start and end)
    current = (archer_pos[0] + step_x, archer_pos[1] + step_y, archer_pos[2] + step_z)
    while current != target_pos:
        # Check if square is occupied
        if cache_manager:
            piece = cache_manager.piece_cache.get(current)
        else:
            piece = board.cache_manager.occupancy.get(current) if cache_manager else board.get_piece(current)

        if piece is not None:
            return False

        # Move to next square
        current = (current[0] + step_x, current[1] + step_y, current[2] + step_z)

    return True


def valid_archery_targets(board: BoardProto, controller: Color,
                         cache_manager: Optional[OptimizedCacheManager] = None) -> List[Tuple[int, int, int]]:
    """
    All enemy pieces within 2-sphere of any friendly ARCHER with clear line of sight.
    """
    out: List[Tuple[int, int, int]] = []
    centres = [
        coord for coord, p in board.list_occupied()
        if p.color == controller and p.ptype == PieceType.ARCHER
    ]
    seen: Set[Tuple[int, int, int]] = set()
    for centre in centres:
        for sq in sphere_centre(board, centre, radius=2):
            if sq in seen:
                continue

            # Use cache manager to get piece at square
            if cache_manager:
                victim = cache_manager.piece_cache.get(sq)
            else:
                victim = board.cache_manager.occupancy.get(sq) if cache_manager else board.get_piece(sq)

            if victim is not None and victim.color != controller:
                # Check line of sight
                if has_line_of_sight(centre, sq, board, cache_manager):
                    out.append(sq)
                    seen.add(sq)
    return out
