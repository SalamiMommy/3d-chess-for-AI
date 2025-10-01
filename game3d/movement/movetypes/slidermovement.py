from __future__ import annotations
"""
# game3d/movement/movetypes/slidermovement.py
Zero-redundancy sliding-move engine — CPU-optimized with Numba.
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange
from numba.typed import List as NbList
from game3d.pieces.enums import Color, PieceType
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import CacheManager
    from game3d.cache.transposition import CompactMove

# --------------------------------------------------------------------------------
# Low-level occupancy helpers
# --------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _occ_code(occ: np.ndarray, x: int, y: int, z: int) -> int:
    return occ[x, y, z]

# --------------------------------------------------------------------------------
# CPU kernel — highly optimized
# --------------------------------------------------------------------------------
@njit(parallel=False, fastmath=True, nogil=False, cache=True)
def _slide_kernel(
    starts: np.ndarray,  # (D,3) int8
    dirs: np.ndarray,    # (D,3) int8
    max_steps: int,
    occ: np.ndarray,     # (9,9,9) uint8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    D = starts.shape[0]
    coords = np.empty((D, max_steps, 3), np.int8)
    valid = np.zeros((D, max_steps), np.bool_)
    hit = np.zeros((D, max_steps), np.uint8)

    for d in prange(D):
        sx, sy, sz = starts[d]
        dx, dy, dz = dirs[d]
        for s in range(1, max_steps + 1):
            tx = sx + dx * s
            ty = sy + dy * s
            tz = sz + dz * s
            if not (0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9):
                break
            coords[d, s - 1] = (tx, ty, tz)
            valid[d, s - 1] = True
            code = _occ_code(occ, tx, ty, tz)
            if code:
                hit[d, s - 1] = code
                break
    return coords, valid, hit

# --------------------------------------------------------------------------------
# Build raw compact data — Numba-optimized, returns list of (fr, to, pt, ic)
# --------------------------------------------------------------------------------
@njit(cache=True, nogil=False)
def _build_compact(
    color_code: int,
    piece_type_value: int,
    start: Tuple[int, int, int],
    coords: np.ndarray,
    valid: np.ndarray,
    hit: np.ndarray,
    allow_capture: bool,
) -> NbList:
    compact = NbList()
    D = coords.shape[0]
    for d in range(D):
        for s in range(coords.shape[1]):
            if not valid[d, s]:
                continue
            tx = coords[d, s, 0]
            ty = coords[d, s, 1]
            tz = coords[d, s, 2]
            to_coord = (tx, ty, tz)
            h = hit[d, s]
            if h == 0:  # empty
                compact.append((start, to_coord, piece_type_value, False))
            else:  # blocked
                if allow_capture and h != color_code:
                    compact.append((start, to_coord, piece_type_value, True))
                break  # stop sliding in this direction
    return compact

# ---------------------------------------------------------------------------
#  Numba expansion – zero-Python-loop, zero-allocation per move
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=False)
def _expand_compact_batch_nb(
    batch: List[Tuple[Tuple[int,int,int],Tuple[int,int,int],bool,bool]]
) -> List[Tuple[Tuple[int,int,int],Tuple[int,int,int],bool]]:
    out = NbList()
    for from_c, to_c, is_cap, _is_promo in batch:
        out.append((from_c, to_c, is_cap))
    return out

def expand_compact_batch(compact_list: List['CompactMove']) -> List[Move]:
    if not compact_list:
        return []
    raw = _expand_compact_batch_nb([cm.unpack() for cm in compact_list])
    return [Move(from_coord=fr, to_coord=to, is_capture=ic, captured_piece=None)
            for fr, to, ic in raw]

# ---------------------------------------------------------------------------
#  Public wrapper – handles raw compact tuples directly
# ---------------------------------------------------------------------------
def _expand_compact(compact: List[Tuple[Tuple[int,int,int], Tuple[int,int,int], int, bool]]) -> List[Move]:
    if not compact:                      # fast-path empty list
        return []
    # Prepare batch with promo=False
    batch = [(fr, to, ic, False) for fr, to, pt, ic in compact]
    # one Numba call – zero Python loops per element
    raw = _expand_compact_batch_nb(batch)
    return [Move(from_coord=fr, to_coord=to, is_capture=ic, captured_piece=None)
            for fr, to, ic in raw]

# --------------------------------------------------------------------------------
# Main generator — CPU-only, highly optimized
# --------------------------------------------------------------------------------
class IntegratedSlidingMovementGenerator:
    __slots__ = ("cache",)

    def __init__(self, cache_manager: 'CacheManager'):
        self.cache = cache_manager

    def generate_sliding_moves(
        self,
        *,
        color: Color,
        piece_type: PieceType,
        position: Tuple[int,int,int],
        directions: np.ndarray,
        max_steps: int = 8,
        allow_capture: bool = True,
        use_symmetry: bool = True,     # <- NEW
        use_amd: bool = True,          # <- NEW
        return_compact: bool = False,
        allow_self_block: bool = False,
    ) -> List[Move] | List['CompactMove']:
        raw_compact = self._compute_compact(
            color, piece_type, position, directions, max_steps,
            allow_capture, use_symmetry, use_amd
        )
        if return_compact:
            from game3d.cache.transposition import CompactMove
            return [
                CompactMove(
                    from_coord=fr,
                    to_coord=to,
                    piece_type=PieceType(pt),
                    is_capture=ic,
                ) for fr, to, pt, ic in raw_compact
            ]
        return _expand_compact(raw_compact)

    def _compute_compact(
        self,
        color: Color,
        piece_type: PieceType,
        pos: Tuple[int,int,int],
        dirs: np.ndarray,
        max_steps: int,
        allow_capture: bool,
        use_symmetry: bool,
        use_amd: bool,
    ) -> List[Tuple[Tuple[int,int,int], Tuple[int,int,int], int, bool]]:
        # ZERO-COPY occupancy view — no expensive export!
        occ = self.cache.piece_cache.get_occupancy_view()

        # Always use Numba CPU kernel — it's faster
        starts = np.repeat(np.array([pos], dtype=np.int8), len(dirs), axis=0)
        coords, valid, hit = _slide_kernel(starts, dirs, max_steps, occ)

        color_code = 1 if color == Color.WHITE else 2
        piece_type_value = piece_type.value  # Assuming PieceType is an IntEnum
        return _build_compact(color_code, piece_type_value, pos, coords, valid, hit, allow_capture)

# --------------------------------------------------------------------------------
# Singleton access
# --------------------------------------------------------------------------------
def get_integrated_movement_generator(cache_manager: 'CacheManager') -> IntegratedSlidingMovementGenerator:
    if not hasattr(cache_manager, "_integrated_movement_gen"):
        cache_manager._integrated_movement_gen = IntegratedSlidingMovementGenerator(cache_manager)
    return cache_manager._integrated_movement_gen

__all__ = ["get_integrated_movement_generator"]
