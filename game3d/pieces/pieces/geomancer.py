# Updated geomancer.py (full file content with fixes)

# game3d/movement/geomancer.py
"""
Unified Geomancer dispatcher
- 1-radius sphere  → walk (normal king move)
- 3-radius surface → block (geomancy effect, no movement, no capture)
"""
from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.kingmovement import generate_king_moves
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.common.common import in_bounds, RADIUS_3_OFFSETS  # ← already pre-computed



if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.cache.cache_manager import OptimizedCacheManager

def block_candidates(
    board: "Board",
    mover_color: "Color",
    cache_manager: "OptimizedCacheManager | None" = None,
) -> List[Tuple[int, int, int]]:
    """
    Return a list of empty squares that the *mover* wants to block
    via geomancy this turn.
    The cache_manager is passed in so you can reuse occupancy data.
    """
    candidates: List[Tuple[int, int, int]] = []

    # --- example logic (tune to your rules) --------------------
    # 1.  Find every friendly Geomancer on the board
    from game3d.common.enums import PieceType
    for sq, piece in board.piece_map.items():
        if piece.color == mover_color and piece.piece_type == PieceType.GEOMANCER:
            # 2.  All empty 3-radius surface squares around that Geomancer
            from game3d.common.common import RADIUS_3_OFFSETS, in_bounds
            x, y, z = sq
            for dx, dy, dz in RADIUS_3_OFFSETS:
                tx, ty, tz = x + dx, y + dy, z + dz
                if not in_bounds((tx, ty, tz)):
                    continue
                if cache_manager is None:          # conservative
                    occ = board.get((tx, ty, tz))
                else:
                    occ = cache_manager.occupancy.get((tx, ty, tz))
                if occ is None:                    # empty → blockable
                    candidates.append((tx, ty, tz))
    # -----------------------------------------------------------
    return candidates
# ------------------------------------------------------------------
# 1.  Core move generator
# ------------------------------------------------------------------
def generate_geomancer_moves(
    cache,          # OptimizedCacheManager
    color: Color,
    x: int, y: int, z: int
) -> List[Move]:
    start = (x, y, z)
    moves: List[Move] = []

    # 1.  Normal king walks
    moves.extend(generate_king_moves(cache, color, x, y, z))

    # 2.  Geomancy block targets (3-sphere surface)
    occ = cache.occupancy
    for dx, dy, dz in RADIUS_3_OFFSETS:          # distance = 3
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_bounds((tx, ty, tz)):
            continue
        if occ.get((tx, ty, tz)) is not None:    # must be empty
            continue

        # Create stationary “effect” move
        mv = convert_legacy_move_args(
            start, start,
            flags=MOVE_FLAGS['GEOMANCY'],
        )
        mv.metadata["is_geomancy_effect"] = True
        mv.metadata["geomancy_target"] = (tx, ty, tz)
        moves.append(mv)

    return moves


# ------------------------------------------------------------------
# 2.  Dispatcher registration
# ------------------------------------------------------------------
@register(PieceType.GEOMANCER)
def geomancer_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_geomancer_moves(state.cache, state.color, x, y, z)


__all__ = ["generate_geomancer_moves"]
