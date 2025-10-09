# networkteleportmoves.py
"""Friendly-Teleporter – teleport to any empty neighbour of a friendly piece
PLUS normal 1-step King moves – all in one jump-batch."""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.movement.movepiece import Move, MOVE_FLAGS, convert_legacy_move_args
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState

# ----------------------------------------------------------
# 1.  26 King directions (re-used here – no import needed)
# ----------------------------------------------------------
_KING_DIRS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
], dtype=np.int8)
# ----------------------------------------------------------
# 2.  Build teleport targets (empty 26-neighbours of any friendly piece)
# ----------------------------------------------------------
def _build_network_directions(cache, color, x, y, z) -> np.ndarray:
    """Return (N,3) int8 array of (target - start) for every valid teleport."""
    start = np.array([x, y, z], dtype=np.int16)
    occ_mask = cache.occupancy.mask          # 9×9×9 bool
    targets = set()

    # all friendly pieces
    for coord, _ in cache.piece_cache.iter_color(color):   # ← coord is a tuple
        fx, fy, fz = coord
        for dx, dy, dz in _KING_DIRS:
            tx, ty, tz = fx + dx, fy + dy, fz + dz
            if in_bounds((tx, ty, tz)) and not occ_mask[tz, ty, tx]:  # ← tuple!
                targets.add((tx, ty, tz))

    targets.discard((x, y, z))          # don't teleport to self
    if not targets:
        return np.empty((0, 3), dtype=np.int8)

    # vectorised (target - start)  →  guaranteed (N,3)
    tarr = np.array(list(targets), dtype=np.int16)
    return (tarr - start).astype(np.int8)
# ----------------------------------------------------------
# 3.  Unified generator – one batch to the jump engine
# ----------------------------------------------------------
def generate_network_teleport_with_king_moves(
    cache, color: Color, x: int, y: int, z: int
) -> List[Move]:
    start = (int(x), int(y), int(z))

    # 1. teleport directions
    tel_dirs = _build_network_directions(cache, color, x, y, z)

    # 2. king directions (cast to same dtype)
    king_dirs = _KING_DIRS.astype(np.int8)

    # 3. concat & deduplicate
    all_dirs = np.unique(np.vstack((tel_dirs, king_dirs)), axis=0)

    # 4. single jump-batch
    jump = get_integrated_jump_movement_generator(cache)
    raw_moves = jump.generate_jump_moves(
        color=color,
        pos=start,
        directions=all_dirs,
        allow_capture=True,  # king captures are legal
    )

    # 5. mark teleports (optional metadata)
    # 5. mark teleports (optional metadata)
    tel_set = {tuple(t) for t in tel_dirs}
    moves = []
    for m in raw_moves:
        mv = convert_legacy_move_args(
                from_coord=start,
                to_coord=m.to_coord,
                is_capture=m.is_capture)
        if tuple(np.array(mv.to_coord) - np.array(start)) in tel_set:
            mv.metadata["is_teleport"] = True
        moves.append(mv)

    return moves          # <-- use this

# ----------------------------------------------------------
# 4.  Dispatcher – state-first
# ----------------------------------------------------------
@register(PieceType.FRIENDLYTELEPORTER)
def friendlytp_move_dispatcher(state: GameState, x: int, y: int, z: int) -> List[Move]:
    return generate_network_teleport_with_king_moves(state.cache, state.color, x, y, z)

__all__ = ["generate_network_teleport_with_king_moves"]
