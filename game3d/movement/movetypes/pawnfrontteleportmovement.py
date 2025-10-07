"""Pawn-Front Teleporter — teleport to empty square directly in front of any enemy pawn.
Zero-redundancy via the jump engine (dynamic directions).
"""
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager as CacheManager

def generate_pawn_front_teleport_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate teleport moves to empty squares directly in front of enemy pawns."""
    start = (x, y, z)

    occ, _ = cache.piece_cache.export_arrays()
    enemy_code = 2 if color == Color.WHITE else 1
    pawn_code = PieceType.PAWN.value | (enemy_code << 3)

    targets: Set[tuple[int, int, int]] = set()

    # scan only occupied squares
    for pos, _ in cache.board.list_occupied():
        px, py, pz = pos
        if occ[px, py, pz] != pawn_code:
            continue
        front = (px, py, pz + 1) if enemy_code == 2 else (px, py, pz - 1)
        if not in_bounds(front):
            continue
        if occ[front] == 0:          # empty
            targets.add(front)

    if not targets:
        return []

    # build dynamic directions array
    dirs = np.array([np.array(t, dtype=np.int8) - np.array(start, dtype=np.int8)
                     for t in targets], dtype=np.int8)

    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=dirs,
        allow_capture=False,   # teleport never captures
                   # keep GPU path
    )
