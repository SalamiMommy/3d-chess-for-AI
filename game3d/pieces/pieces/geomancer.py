# Updated geomancer.py (with consistent cache access)

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.common.coord_utils import in_bounds
from game3d.common.constants import RADIUS_3_OFFSETS
from game3d.movement.cache_utils import get_occupancy_safe, ensure_int_coords

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState  # Added missing import

def block_candidates(
    board: "Board",
    mover_color: "Color",
    cache_manager: "OptimizedCacheManager | None" = None,
) -> List[Tuple[int, int, int]]:
    """
    Return empty squares that <mover_color> may block via geomancy this turn.
    """
    if cache_manager is None:
        cache_manager = board.cache_manager
    candidates: List[Tuple[int, int, int]] = []

    # Use standardized cache iteration
    for sq, piece in cache_manager.get_pieces_of_color(mover_color):
        if piece.ptype is not PieceType.GEOMANCER:
            continue

        # Collect empty 3-radius surface squares around that Geomancer
        x, y, z = sq
        for dx, dy, dz in RADIUS_3_OFFSETS:
            tx, ty, tz = x + dx, y + dy, z + dz
            if not in_bounds((tx, ty, tz)):
                continue

            # Use standardized occupancy check
            if get_occupancy_safe(cache_manager, (tx, ty, tz)) is None:
                candidates.append((tx, ty, tz))

    return candidates

def generate_geomancer_moves(
    cache: 'OptimizedCacheManager',
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    start = (x, y, z)
    moves: List[Move] = []

    # 1. Normal king walks
    moves.extend(generate_king_moves(cache, color, x, y, z))

    # 2. Geomancy block targets (3-sphere surface)
    for dx, dy, dz in RADIUS_3_OFFSETS:
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_bounds((tx, ty, tz)):
            continue

        # Use standardized occupancy check
        if get_occupancy_safe(cache, (tx, ty, tz)) is not None:
            continue

        # Create stationary "effect" move
        mv = convert_legacy_move_args(
            start, start,
            flags=MOVE_FLAGS['GEOMANCY'],
        )
        mv.metadata["is_geomancy_effect"] = True
        mv.metadata["geomancy_target"] = (tx, ty, tz)
        moves.append(mv)

    return moves

@register(PieceType.GEOMANCER)
def geomancer_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    x, y, z = ensure_int_coords(x, y, z)
    return generate_geomancer_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_geomancer_moves"]
