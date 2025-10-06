from __future__ import annotations
"""
game3d/movement/movetypes/jumpmovement.py
Ultra-fast JUMP-move engine for 9x9x9 chess. Optimized for batch directions and minimal Python overhead.
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds
from game3d.movement.movepiece import MOVE_FLAGS
if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager

# ------------------------------------------------------------------
#  Tiny helpers
# ------------------------------------------------------------------
@njit(cache=True, inline="always")
def _occ_code(occ: np.ndarray, x: int, y: int, z: int) -> int:
    return occ[x, y, z]

@njit(cache=True, inline="always")
def _in_bounds(x: int, y: int, z: int) -> bool:
    return 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9

# ------------------------------------------------------------------
#  Optimized Kernel – outputs flat arrays for batch Move creation
# ------------------------------------------------------------------
@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _jump_kernel_direct(
    start: Tuple[int, int, int],
    dirs: np.ndarray,
    occ: np.ndarray,
    own_code: int,
    enemy_code: int,
    allow_capture: bool,
    enemy_has_priests: bool,
):
    # Preallocate output arrays (max possible size)
    n = dirs.shape[0]
    out_coords = np.empty((n, 3), dtype=np.int16)
    out_captures = np.zeros(n, dtype=np.bool_)
    out_count = 0

    sx, sy, sz = start
    for d in prange(n):
        dx, dy, dz = dirs[d]
        tx = sx + dx
        ty = sy + dy
        tz = sz + dz
        if not _in_bounds(tx, ty, tz):
            continue
        h = occ[tx, ty, tz]
        if h == 0:
            out_coords[out_count] = (tx, ty, tz)
            out_captures[out_count] = False
            out_count += 1
        elif allow_capture and h != own_code:
            if h == enemy_code and enemy_has_priests:
                continue
            out_coords[out_count] = (tx, ty, tz)
            out_captures[out_count] = True
            out_count += 1

    # Slice to used portion
    return out_coords[:out_count], out_captures[:out_count]

# ------------------------------------------------------------------
#  Batch Move creation — uses MovePool for low overhead
# ------------------------------------------------------------------
def _build_jump_moves(
    color: Color,
    ptype: PieceType,
    start: Tuple[int, int, int],
    coords: np.ndarray,
    captures: np.ndarray
) -> List[Move]:
    # Use object pool for fast allocation, batch creation is much faster
    return Move.create_batch(start, coords, captures)

# ------------------------------------------------------------------
#  Main generator – only CPU path (no GPU, no loops)
# ------------------------------------------------------------------
class IntegratedJumpMovementGenerator:
    __slots__ = ("cache",)

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def generate_jump_moves(
        self,
        *,
        color: Color,
        pos: Tuple[int, int, int],
        directions: np.ndarray,
        allow_capture: bool = True,
        use_amd: bool = True,  # ignored – CPU is faster
    ) -> List[Move]:
        occ, _ = self.cache.piece_cache.export_arrays()
        own_code = 1 if color == Color.WHITE else 2
        enemy_code = PieceType.KING.value | ((3 - own_code) << 3)
        enemy_has_priests = self._enemy_still_has_priests(color)

        coords, captures = _jump_kernel_direct(
            pos,
            directions.astype(np.int16),
            occ,
            own_code,
            enemy_code,
            allow_capture,
            enemy_has_priests,
        )
        return _build_jump_moves(color, PieceType.PAWN, pos, coords, captures)

    def _enemy_still_has_priests(self, color: Color) -> bool:
        occ, piece_array = self.cache.piece_cache.export_arrays()
        enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        priest_code = PieceType.PRIEST.value | ((1 if enemy_color == Color.WHITE else 2) << 3)
        return np.any(piece_array == priest_code)

# ------------------------------------------------------------------
#  Singleton access
# ------------------------------------------------------------------
def get_integrated_jump_movement_generator(cache_manager: CacheManager) -> IntegratedJumpMovementGenerator:
    if not hasattr(cache_manager, "_integrated_jump_gen"):
        cache_manager._integrated_jump_gen = IntegratedJumpMovementGenerator(cache_manager)
    return cache_manager._integrated_jump_gen

__all__ = ["get_integrated_jump_movement_generator"]
