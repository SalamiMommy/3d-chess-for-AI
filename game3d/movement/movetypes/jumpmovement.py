from __future__ import annotations
"""
game3d/movement/movetypes/jumpmovement.py
Zero-redundancy JUMP-move engine – 5600-X optimised, GPU code removed.
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds

if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager

# ------------------------------------------------------------------
#  Tiny helpers
# ------------------------------------------------------------------
@njit(cache=True, inline="always")
def _occ_code(occ: np.ndarray, x: int, y: int, z: int) -> int:
    return occ[x, y, z]

# ------------------------------------------------------------------
#  Inline alias for Numba – keeps same logic, zero call overhead
# ------------------------------------------------------------------
@njit(cache=True, inline="always")
def _in_bounds(x: int, y: int, z: int) -> bool:
    return 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9

# ------------------------------------------------------------------
#  Kernel – use the local alias
# ------------------------------------------------------------------
@njit(parallel=False, fastmath=True, nogil=False, cache=True)
def _jump_kernel_direct(
    start: Tuple[int, int, int],
    dirs: np.ndarray,
    occ: np.ndarray,
    own_code: int,
    enemy_code: int,
    allow_capture: bool,
    enemy_has_priests: bool,
) -> List[Tuple[int, int, int, bool]]:
    out: List[Tuple[int, int, int, bool]] = []
    sx, sy, sz = start
    for d in prange(dirs.shape[0]):
        dx, dy, dz = dirs[d]
        tx = sx + dx
        ty = sy + dy
        tz = sz + dz
        if not _in_bounds(tx, ty, tz):          # <-- local alias
            continue
        h = occ[tx, ty, tz]
        if h == 0:
            out.append((tx, ty, tz, False))
        elif allow_capture and h != own_code:
            if h == enemy_code and enemy_has_priests:
                continue
            out.append((tx, ty, tz, True))
    return out

# ------------------------------------------------------------------
#  Public wrapper – zero Python loops
# ------------------------------------------------------------------
def _build_jump_moves(
    color: Color,
    piece_type: PieceType,
    start: Tuple[int, int, int],
    raw: List[Tuple[int, int, int, bool]],
) -> List[Move]:
    return [
        Move(
            from_coord=start,
            to_coord=(x, y, z),
            is_capture=is_cap,
            captured_piece=None,
        )
        for x, y, z, is_cap in raw
    ]

# ------------------------------------------------------------------
#  Main generator – CUDA path removed
# ------------------------------------------------------------------
class IntegratedJumpMovementGenerator:
    __slots__ = ("cache",)

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def generate_jump_moves(
        self,
        *,
        color: Color,
        position: Tuple[int, int, int],
        directions: np.ndarray,
        allow_capture: bool = True,
        use_amd: bool = True,  # ignored – CPU is faster
    ) -> List[Move]:
        occ, _ = self.cache.piece_cache.export_arrays()
        own_code = 1 if color == Color.WHITE else 2
        enemy_code = PieceType.KING.value | ((3 - own_code) << 3)
        enemy_has_priests = self._enemy_still_has_priests(color)

        raw = _jump_kernel_direct(
            position,
            directions.astype(np.int16),
            occ,
            own_code,
            enemy_code,
            allow_capture,
            enemy_has_priests,
        )
        return _build_jump_moves(color, PieceType.PAWN, position, raw)

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
