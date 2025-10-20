# jumpmovement.py
from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit

from game3d.common.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import in_bounds_scalar, filter_valid_coords, color_to_code

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager


# ----------  JIT kernels stay exactly the same  ----------
@njit(cache=True, inline="always")
def _in_bounds(x: int, y: int, z: int) -> bool:
    return 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9


@njit(fastmath=True, cache=True)
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
    for d in range(dirs.shape[0]):
        dx, dy, dz = dirs[d, 0], dirs[d, 1], dirs[d, 2]
        tx, ty, tz = sx + dx, sy + dy, sz + dz
        if not _in_bounds(tx, ty, tz):
            continue
        h = occ[tz, ty, tx]
        if h == 0:
            out.append((tx, ty, tz, False))
        elif allow_capture and h != own_code:
            if h == enemy_code and enemy_has_priests:
                continue
            out.append((tx, ty, tz, True))
    return out


# ----------  helper that builds Move objects  ----------
def _build_jump_moves(
    color: Color,
    ptype: PieceType,
    start: Tuple[int, int, int],
    raw: List[Tuple[int, int, int, bool]],
) -> List[Move]:
    if not raw:
        return []
    if any(c < 0 or c >= 9 for c in start):
        print(f"[ERROR] _build_jump_moves: invalid start position {start}")
        return []

    to_coords = np.array([[x, y, z] for x, y, z, _ in raw], dtype=np.int32)
    to_coords = [tuple(int(c) for c in row) for row in
                filter_valid_coords(to_coords, log_oob=True)]
    valid_raw = [
        (int(x), int(y), int(z), is_cap)
        for (x, y, z), (_, _, _, is_cap) in zip(to_coords, raw)
        if 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9
    ]

    rejected = len(raw) - len(valid_raw)
    if rejected:
        print(f"[WARNING] _build_jump_moves rejected {rejected} OOB coords from {start}")

    if not valid_raw:
        return []

    n = len(valid_raw)
    to_coords = np.empty((n, 3), dtype=np.int32)
    captures = np.empty(n, dtype=bool)
    for i, (x, y, z, is_cap) in enumerate(valid_raw):
        to_coords[i] = (x, y, z)
        captures[i] = is_cap

    try:
        return Move.create_batch(start, to_coords, captures)
    except (IndexError, KeyError) as e:
        print(f"[ERROR] Move.create_batch failed for {start}: {e}")
        return []


# ----------  main generator class  ----------
class IntegratedJumpMovementGenerator:
    __slots__ = ("mgr", "_jumptables", "_priest_cache")

    def __init__(self, manager: OptimizedCacheManager) -> None:
        self.mgr = manager
        self._jumptables: dict = {}
        self._priest_cache: dict = {}

    def _get_occ_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (colour_codes, piece_type_codes) as **copies** for Numba."""
        return self.mgr.occupancy._occ.copy(), self.mgr.occupancy._ptype.copy()

    def generate_jump_moves(
        self,
        *,
        color: Color,
        pos: Tuple[int, int, int],
        directions: np.ndarray,
        allow_capture: bool = True,
        use_amd: bool = True,
        piece_name: str | None = None,
    ) -> List[Move]:
        occ, _ = self._get_occ_arrays()  # get arrays through manager
        own_code = color_to_code(color)
        enemy_code = 1 if color == Color.BLACK else 2  # inverse colour code

        enemy_has_priests = self._enemy_still_has_priests_fast(color)

        # CORRECTED: Use manager method
        is_buffed = self.mgr.is_movement_buffed(pos, color)

        raw = _jump_kernel_direct(
            pos,
            directions.astype(np.int16),
            occ,
            own_code,
            enemy_code,
            allow_capture,
            enemy_has_priests,
        )

        standard_moves = _build_jump_moves(color, PieceType.PAWN, pos, raw)

        if not is_buffed:
            return standard_moves

        signs = np.sign(directions).astype(np.int16)
        extended_dirs = directions.astype(np.int16) + signs
        dest_coords = np.array(pos, dtype=np.int16) + extended_dirs
        extended_dirs = filter_valid_coords(extended_dirs)  # keeps (N,3) shape

        if len(extended_dirs) == 0:
            return standard_moves

        extended_raw = _jump_kernel_direct(
            pos,
            extended_dirs,
            occ,
            own_code,
            enemy_code,
            allow_capture,
            enemy_has_priests,
        )
        extended_moves = _build_jump_moves(color, PieceType.PAWN, pos, extended_raw)

        buffed_flag = MOVE_FLAGS["BUFFED"] << 20
        for m in extended_moves:
            m._data |= buffed_flag
            m._cached_hash = None

        return standard_moves + extended_moves

    def _enemy_still_has_priests_fast(self, color: Color) -> bool:
        # CORRECTED: Use manager method (has_priest is a manager method)
        return self.mgr.has_priest(color.opposite())


# ----------  factory kept for compatibility  ----------
def get_integrated_jump_movement_generator(cm: OptimizedCacheManager) -> IntegratedJumpMovementGenerator:
    if not hasattr(cm, "_integrated_jump_gen") or cm._integrated_jump_gen is None:
        cm._integrated_jump_gen = IntegratedJumpMovementGenerator(cm)
    return cm._integrated_jump_gen


__all__ = ["get_integrated_jump_movement_generator"]
