# game3d/movement/movetypes/swapmovement.py

from __future__ import annotations

from typing import List
import numpy as np

from game3d.pieces.enums import Color
from game3d.movement.movetypes.jumpmovement import get_integrated_jump_movement_generator
from game3d.cache.manager import CacheManager

def generate_swapper_moves(
    cache: CacheManager,
    color: Color,
    x: int, y: int, z: int
) -> List['Move']:
    """Swap with any friendly piece using the jump engine."""
    start = (x, y, z)

    # 1. Export occupancy (color-only: 0=empty, 1=white, 2=black)
    occ, _ = cache.piece_cache.export_arrays()
    own_code = 1 if color == Color.WHITE else 2

    # Validate starting square
    if occ[x, y, z] != own_code:
        return []

    # 2. Find all other friendly pieces
    targets: List[Tuple[int, int, int]] = []
    for tx in range(9):
        for ty in range(9):
            for tz in range(9):
                if (tx, ty, tz) == start:
                    continue
                if occ[tx, ty, tz] == own_code:
                    targets.append((tx, ty, tz))

    if not targets:
        return []

    # 3. Build directions (use int16 to avoid overflow in later indexing)
    start_arr = np.array(start, dtype=np.int16)
    dirs = np.empty((len(targets), 3), dtype=np.int16)
    for i, t in enumerate(targets):
        dirs[i] = np.array(t, dtype=np.int16) - start_arr

    # 4. Use jump generator (no capture)
    gen = get_integrated_jump_movement_generator(cache)
    moves = gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=dirs,
        allow_capture=False,
        
    )

    # 5. Mark as swap (optional)
    for m in moves:
        m.is_swap = True
    return moves
