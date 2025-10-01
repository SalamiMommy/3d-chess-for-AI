"""Network Teleporter — teleport to any empty square adjacent to any friendly piece.
Zero-redundancy via the jump engine (dynamic directions list).
"""

from __future__ import annotations

from typing import List, Set
import numpy as np

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.cache.manager import CacheManager
from game3d.common.common import in_bounds

# 26 3-D neighbour deltas
_NEIGHBOR_DIRECTIONS = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == dy == dz == 0)
], dtype=np.int8)

def generate_network_teleport_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Generate all legal network-teleport moves from (x, y, z)."""
    start = (x, y, z)

    # --- 1. collect candidate targets ---
    occ, piece_array = cache.piece_cache.export_arrays()
    own_code = 1 if color == Color.WHITE else 2
    candidates: Set[tuple[int, int, int]] = set()

    # iterate only over *occupied* squares
    for pos, _ in cache.board.list_occupied():
        px, py, pz = pos
        if occ[px, py, pz] != own_code:        # not friendly
            continue
        for dx, dy, dz in _NEIGHBOR_DIRECTIONS:
            tx, ty, tz = px + dx, py + dy, pz + dz
            if not in_bounds((tx, ty, tz)):
                continue
            if occ[tx, ty, tz] == 0:           # empty
                candidates.add((tx, ty, tz))

    # --- 2. convert to directions for the jump engine ---
    if not candidates:
        return []
    dirs = np.array([np.array(t, dtype=np.int8) - np.array(start, dtype=np.int8)
                     for t in candidates], dtype=np.int8)

    # --- 3. hand off to the existing generator ---
    gen = get_integrated_jump_movement_generator(cache)
    return gen.generate_jump_moves(
        color=color,
        position=start,
        directions=dirs,
        allow_capture=False,   # network teleport never captures
        use_amd=True           # keep GPU path
    )
