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
    """
    Optimized: Generate all legal swapper moves (swap with any friendly piece) using batch numpy logic.
    """
    start = (x, y, z)

    # Export occupancy array: occ[x, y, z] == 1 (white), 2 (black), 0 (empty)
    occ, _ = cache.piece_cache.export_arrays()
    own_code = 1 if color == Color.WHITE else 2

    # Validate starting square is occupied by own piece
    if occ[x, y, z] != own_code:
        return []

    # Find all friendly piece coordinates (excluding self) in one batch
    coords = np.stack(np.where(occ == own_code), axis=-1)  # shape (N, 3), order: z, y, x
    if coords.shape[0] == 0:
        return []

    # Convert to (x, y, z) order
    coords_xyz = coords[:, [2, 1, 0]]
    # Exclude starting position
    mask_not_self = ~np.all(coords_xyz == np.array(start), axis=1)
    targets = coords_xyz[mask_not_self]

    if targets.shape[0] == 0:
        return []

    # Batch direction calculation: targets - start
    start_array = np.array(start, dtype=np.int16)
    dirs = targets.astype(np.int16) - start_array

    # Use jump generator (no capture)
    gen = get_integrated_jump_movement_generator(cache)
    moves = gen.generate_jump_moves(
        color=color,
        pos=start,
        directions=dirs,
        allow_capture=False,
    )

    # Mark as swap (optional metadata for downstream use)
    for m in moves:
        m.is_swap = True

    return moves

    # 5. Mark as swap (optional)
    for m in moves:
        m.is_swap = True
    return moves
