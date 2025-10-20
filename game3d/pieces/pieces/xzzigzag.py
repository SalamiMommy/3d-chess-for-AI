"""
XZ-Zig-Zag — 9-step zig-zag rays in XZ-plane + dispatcher (consolidated).
Exports:
  generate_xz_zigzag_moves(cache, color, x, y, z) -> list[Move]
  (decorated) xz_zigzag_dispatcher(state, x, y, z) -> list[Move]
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from game3d.common.enums import Color, PieceType
from game3d.movement.registry import register
from game3d.movement.movetypes.slidermovement import generate_slider_moves_kernel
from game3d.movement.movepiece import Move, MOVE_FLAGS

# ----------------------------------------------------------
# 1.  Pre-computed zig-zag vectors (same table as before)
# ----------------------------------------------------------
def _build_xz_zigzag_vectors() -> np.ndarray:
    vecs = []
    for pri, sec in ((1, -1), (-1, 1)):
        seq = []
        curr = np.zeros(3, dtype=np.int8)
        move_primary = True
        for seg in range(3):                # 3 segments × 3 steps
            step = np.zeros(3, dtype=np.int8)
            ax = 0 if move_primary else 2  # Fixed: XZ-plane uses axes 0 (X) and 2 (Z)
            step[ax] = pri if move_primary else sec
            for _ in range(3):
                curr += step
                seq.append(curr.copy())
            move_primary ^= 1
        vecs.extend(seq)
    return np.array(vecs, dtype=np.int8)

XZ_ZIGZAG_DIRECTIONS = _build_xz_zigzag_vectors()

# ----------------------------------------------------------
# 2.  Public generator (drop-in replacement)
# ----------------------------------------------------------
def generate_xz_zigzag_moves(cache: CacheManager,
                             color: Color,
                             x: int, y: int, z: int) -> List[Move]:
    """Generate XZ-zig-zag moves using only the slider kernel."""
    raw = generate_slider_moves_kernel(
        pos=(x, y, z),
        directions=XZ_ZIGZAG_DIRECTIONS,
        occupancy=cache.occupancy._occ,  # ✅ Fixed: Remove .mask
        color=color.value,
        max_distance=16
    )
    # Convert raw tuples → Move objects
    return [
        Move(from_coord=(x, y, z), to_coord=(nx, ny, nz),
             flags=MOVE_FLAGS['CAPTURE'] if is_cap else 0)
        for nx, ny, nz, is_cap in raw
    ]

# ----------------------------------------------------------
# 3.  Dispatcher (registered here)
# ----------------------------------------------------------
@register(PieceType.XZZIGZAG)
def xz_zigzag_move_dispatcher(state, x: int, y: int, z: int) -> List[Move]:
    return generate_xz_zigzag_moves(state.cache, state.color, x, y, z)

__all__ = ['generate_xz_zigzag_moves']
