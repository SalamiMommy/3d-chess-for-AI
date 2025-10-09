from __future__ import annotations
"""
game3d/movement/movetypes/jumpmovement.py
Zero-redundancy jump-move engine — now supports disk-backed precomputed jump tables, with fallback.
"""

from typing import List, Tuple, TYPE_CHECKING
import os
import numpy as np
from numba import njit, prange

from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move, MOVE_FLAGS
from game3d.common.common import coord_to_idx, in_bounds
from game3d.movement.movepiece import MOVE_FLAGS
if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager

# ------------------------------------------------------------------
#  Precomputed jump table loader
# ------------------------------------------------------------------
_PRECOMPUTED_DIR = os.path.join(os.path.dirname(__file__), "precomputed")

def load_precomputed_jumptable(piece_name: str):
    """Load precomputed jump table for a piece type from disk, or return None if unavailable."""
    filename = os.path.join(_PRECOMPUTED_DIR, f"{piece_name}_jumptable.npy")
    if not os.path.isfile(filename):
        return None
    arr = np.load(filename, allow_pickle=True)
    if arr.shape[0] != 729:
        return None
    return arr

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
    ptype: PieceType,
    start: Tuple[int, int, int],
    raw: List[Tuple[int, int, int, bool]],
) -> List[Move]:
    if not raw:
        return []

    # Convert to numpy for batch creation
    to_coords = np.array([(x, y, z) for x, y, z, _ in raw], dtype=np.int32)
    captures = np.array([is_cap for _, _, _, is_cap in raw], dtype=bool)

    return Move.create_batch(start, to_coords, captures)

# ------------------------------------------------------------------
#  Main generator – supports jump table fallback
# ------------------------------------------------------------------
class IntegratedJumpMovementGenerator:
    __slots__ = ("cache", "_jumptables")

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self._jumptables = {}

    def _get_precomputed_moves(self, piece_name: str, pos: Tuple[int, int, int], color: Color, allow_capture=True):
        """Return Move objects for all legal jumps from precomputed table, after occupancy filtering."""
        if piece_name not in self._jumptables:
            self._jumptables[piece_name] = load_precomputed_jumptable(piece_name)
        table = self._jumptables[piece_name]
        if table is None:
            return None
        idx = coord_to_idx(pos)
        raw_destinations = table[idx]  # List of (x, y, z) tuples

        occ, _ = self.cache.piece_cache.export_arrays()
        own_code = 1 if color == Color.WHITE else 2
        enemy_code = PieceType.KING.value | ((3 - own_code) << 3)
        enemy_has_priests = self._enemy_still_has_priests(color)

        moves = []
        for tx, ty, tz in raw_destinations:
            if not _in_bounds(tx, ty, tz):
                continue
            h = occ[tx, ty, tz]
            is_cap = False
            if h == 0:
                is_cap = False
            elif allow_capture and h != own_code:
                if h == enemy_code and enemy_has_priests:
                    continue
                is_cap = True
            else:
                continue  # Blocked by friendly
            moves.append(Move.create_simple(
                pos, (tx, ty, tz),
                is_capture=is_cap
            ))
        return moves

    def generate_jump_moves(
        self,
        *,
        color: Color,
        pos: Tuple[int, int, int],
        directions: np.ndarray,
        allow_capture: bool = True,
        use_amd: bool = True,  # ignored – CPU is faster
        piece_name: str = None,  # Optional: for precomputed lookup
    ) -> List[Move]:
        # Try precomputed first if piece_name is given
        if piece_name:
            moves = self._get_precomputed_moves(piece_name, pos, color, allow_capture=allow_capture)
            if moves is not None:
                return moves

        # Fallback: use kernel
        occ, _ = self.cache.piece_cache.export_arrays()
        own_code = 1 if color == Color.WHITE else 2
        enemy_code = PieceType.KING.value | ((3 - own_code) << 3)
        enemy_has_priests = self._enemy_still_has_priests(color)

        raw = _jump_kernel_direct(
            pos,
            directions.astype(np.int16),
            occ,
            own_code,
            enemy_code,
            allow_capture,
            enemy_has_priests,
        )
        return _build_jump_moves(color, PieceType.PAWN, pos, raw)

    def _enemy_still_has_priests(self, color: Color) -> bool:
        occ, piece_array = self.cache.piece_cache.export_arrays()
        enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        priest_code = PieceType.PRIEST.value | ((1 if enemy_color == Color.WHITE else 2) << 3)
        return np.any(piece_array == priest_code)

def dirty_squares_jump(
    mv: Move,
    mover: Color,
    cache_manager: CacheManager
) -> set[Tuple[int,int,int]]:
    """
    Return the coordinates whose jump attack set is *possibly* different
    after this move.  Over-approximation is fine.
    """
    dirty: set[Tuple[int,int,int]] = set()

    # 1.  Piece that just moved
    dirty.add(mv.from_coord)          # old location – now empty
    dirty.add(mv.to_coord)            # new location – now occupied

    # 2.  Captured piece (if any) could have been a jumper
    if mv.is_capture:
        dirty.add(mv.to_coord)

    # 3.  Jump pieces are never blocked → no ray scan needed
    # 4.  King-jump or priest-jump aura may affect neighbours – add 1-ring
    for dx,dy,dz in product((-1,0,1), repeat=3):
        if dx==dy==dz==0:
            continue
        dirty.add((mv.from_coord[0]+dx, mv.from_coord[1]+dy, mv.from_coord[2]+dz))
        dirty.add((mv.to_coord[0]+dx,   mv.to_coord[1]+dy,   mv.to_coord[2]+dz))

    # clamp to board
    return {c for c in dirty if in_bounds(*c)}
# ------------------------------------------------------------------
#  Singleton access
# ------------------------------------------------------------------
def get_integrated_jump_movement_generator(cm: CacheManager) -> IntegratedJumpMovementGenerator:
    if cm._integrated_jump_gen is None:
        #  create AND assign in one step
        cm._integrated_jump_gen = IntegratedJumpMovementGenerator(cm)
    return cm._integrated_jump_gen

__all__ = ["get_integrated_jump_movement_generator"]
