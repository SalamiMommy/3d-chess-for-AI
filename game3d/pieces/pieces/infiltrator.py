# infiltrator.py
from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING
from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move
from game3d.common.coord_utils import in_bounds

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_KING_DIRS = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int8,
)


def _enemy_pawn_front_squares(mgr: OptimizedCacheManager, enemy_color: Color) -> List[Tuple[int, int, int]]:
    """Return empty squares directly in front of every enemy pawn."""
    occ = mgr.occupancy._occ  # 0…2 colour codes
    fronts: list[tuple[int, int, int]] = []

    # iterate through manager → occupancy → iter_color
    for pos, piece in mgr.occupancy.iter_color(enemy_color):
        if piece.ptype != PieceType.PAWN:
            continue
        # Front direction depends on enemy color (BLACK moves +z, WHITE moves -z)
        front = (pos[0], pos[1], pos[2] + (1 if enemy_color == Color.BLACK else -1))
        if in_bounds(front) and occ[front] == 0:
            fronts.append(front)

    return fronts


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_infiltrator_moves(cache, color, x, y, z) -> List[Move]:
    """Generate all pseudo-legal moves for an Infiltrator."""
    jump_gen = get_integrated_jump_movement_generator(cache)

    # 1. Teleport targets -> directions (guarantee (N,3) shape)
    enemy_color = color.opposite()
    teleport_targets = _enemy_pawn_front_squares(cache, enemy_color)

    if not teleport_targets:                       # empty list -> empty 0×3 array
        teleport_dirs = np.empty((0, 3), dtype=np.int8)
    else:
        start = np.array([x, y, z], dtype=np.int16)
        targets = np.array(teleport_targets, dtype=np.int16)
        teleport_dirs = (targets - start).astype(np.int8)

    # 2. Produce moves via the shared jump kernel
    moves = jump_gen.generate_jump_moves(
        color=color, pos=(x, y, z), directions=teleport_dirs
    )
    moves += jump_gen.generate_jump_moves(
        color=color, pos=(x, y, z), directions=_KING_DIRS
    )

    return moves
# ---------------------------------------------------------------------------
# Dispatcher entry-point
# ---------------------------------------------------------------------------
@register(PieceType.INFILTRATOR)
def infiltrator_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_infiltrator_moves(state.cache, state.color, x, y, z)


__all__ = ["generate_infiltrator_moves"]
