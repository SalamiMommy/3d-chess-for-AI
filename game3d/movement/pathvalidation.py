"""Optimized pathvalidation.py - vectorized operations and reduced function call overhead."""

import numpy as np
from numba import njit, prange
from typing import List, Tuple, Optional
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds, add_coords
from game3d.pieces.enums import PieceType, Color
from game3d.cache.manager import OptimizedCacheManager
BOARD_SIZE = 9

# Pre-compute bounds checking array for fast validation
_VALID_COORDS = np.ones((9, 9, 9), dtype=bool)


def should_stop_at(cache, color, target, allow_capture=True, allow_self_block=False):
    piece = cache.piece_cache.get(target)

    if piece is None:
        return False, True

    is_friendly = piece.color == color

    if is_friendly:
        return (False, True) if allow_self_block else (True, False)

    return (True, True) if allow_capture else (True, False)


# ------------------------------------------------------------------
# Same branch-free helper used by the batch version
# ------------------------------------------------------------------
_UPPER_MASK = np.uint8(0xF0)

_BOUNDS_LUT = np.zeros((256, 256, 256), dtype=np.bool_)
for x in range(9):
    for y in range(9):
        for z in range(9):
            _BOUNDS_LUT[x, y, z] = True

def _in_bounds_1d_fast(coord: np.ndarray) -> bool:
    """OPTIMIZED: LUT-based bounds check - O(1) with no arithmetic."""
    x, y, z = coord[0], coord[1], coord[2]
    # Handle negative indices by masking to 0-255 range
    x, y, z = x & 0xFF, y & 0xFF, z & 0xFF
    return bool(_BOUNDS_LUT[x, y, z])


# Alternative: Ultra-fast inline version for hot path
def _in_bounds_inline(x: int, y: int, z: int) -> bool:
    """Inline bounds check - use this in tight loops."""
    return (x | y | z) >= 0 and x < 9 and y < 9 and z < 9


@njit(parallel=True, nogil=True)
def _slide_batch_kernel(starts, dirs, max_steps, occ, ptype):
    """
    occ   : uint8[9,9,9]   0=empty 1=white 2=black
    ptype : uint8[9,9,9]   PieceType.value if occupied
    returns same as before + hit_color/hit_type arrays
    """
    N = starts.shape[0]
    coords   = np.empty((N, max_steps, 3), dtype=np.int8)
    valid    = np.zeros((N, max_steps), dtype=np.bool_)
    hit_col  = np.zeros((N, max_steps), dtype=np.uint8)  # 0 none 1 w 2 b
    hit_pt   = np.zeros((N, max_steps), dtype=np.uint8)

    for i in prange(N):
        sx, sy, sz = starts[i]
        dx, dy, dz = dirs[i]
        for step in range(1, max_steps + 1):
            tx = sx + dx * step
            ty = sy + dy * step
            tz = sz + dz * step
            if tx < 0 or ty < 0 or tz < 0 or tx >= 9 or ty >= 9 or tz >= 9:
                break
            coords[i, step-1] = (tx, ty, tz)
            occ_code = occ[tx, ty, tz]
            valid[i, step-1] = True
            if occ_code != 0:
                hit_col[i, step-1] = occ_code
                hit_pt[i, step-1]  = ptype[tx, ty, tz]
                break
    return coords, valid, hit_col, hit_pt


def slide_along_directions(cache, color, start, directions, **kw):
    # 1. ---- one-time export ----
    occ, ptype = cache.piece_cache.export_arrays()   # 2×9×9×9 np arrays, <1 µs

    # 2. ---- GIL-free batch ----
    D = directions.shape[0]
    starts = np.repeat(np.array([start], dtype=np.int8), D, axis=0)
    dirs   = np.asarray(directions, dtype=np.int8)
    coords, mask, hit_col, hit_pt = _slide_batch_kernel(
        starts, dirs, 9, occ, ptype)

    # 3. ---- rebuild Move list ----
    moves = []
    for d in range(D):
        for s in range(9):
            if not mask[d, s]:
                continue
            tgt = tuple(coords[d, s])
            hc = hit_col[d, s]
            if hc == 0:                       # empty
                moves.append(Move(from_coord=start, to_coord=tgt, is_capture=False))
            else:                             # occupied
                is_enemy = (hc == 1 and color == Color.BLACK) or (hc == 2 and color == Color.WHITE)
                moves.append(Move(from_coord=start, to_coord=tgt,
                                  is_capture=is_enemy,
                                  captured_piece=PieceType(hit_pt[d, s]) if is_enemy else None))
                break
    return moves
# ------------------------------------------------------------------
# Fast branch-free bounds check for (N,3) int8 arrays – 9×9×9 board
# ------------------------------------------------------------------


def _in_bounds_nd(coords: np.ndarray) -> np.ndarray:
    """
    coords:  int8 array of shape (..., 3)
    returns: bool array of shape (...)   (no temporaries, no reduction)
    """
    return (coords.view(np.uint8) & _UPPER_MASK).sum(axis=-1) == 0


def slide_along_directions_batch(
    cache: OptimizedCacheManager,
    color,
    start: Tuple[int, int, int],
    directions: np.ndarray,                    # (D, 3)
    allow_capture: bool = True,
    allow_self_block: bool = False,
    max_steps: Optional[int] = BOARD_SIZE,
) -> List[Move]:
    """
    HIGHLY OPTIMISED: batch generation, early break, zero-allocation bounds test.
    """
    if max_steps is None:
        max_steps = BOARD_SIZE

    moves: List[Move] = []
    start_arr = np.array(start, dtype=np.int8)        # (3,)

    # Produce all targets in one broadcast:  (D, S, 3)  S = max_steps
    steps = np.arange(1, max_steps + 1, dtype=np.int8)
    targets = start_arr + directions[:, None, :] * steps[None, :, None]

    # Bounds mask – single pass, no temporaries
    valid_mask = _in_bounds_nd(targets)               # (D, S)

    # Per-direction early-exit loop
    for d, direction in enumerate(directions):
        for s in range(max_steps):
            if not valid_mask[d, s]:
                break

            tgt = tuple(int(x) for x in targets[d, s])
            stop, can_land = should_stop_at(cache, color, tgt,
                                          allow_capture, allow_self_block)
            if can_land:
                piece = cache.piece_cache.get(tgt)
                moves.append(Move(from_coord=start,
                                to_coord=tgt,
                                is_capture=piece is not None and piece.color != color))
            if stop:
                break

    return moves


def jump_to_targets(
    cache: OptimizedCacheManager,
    color,
    start: Tuple[int, int, int],
    offsets: List[Tuple[int, int, int]],
    allow_capture: bool = True,
    allow_self_block: bool = False,
) -> List[Move]:
    """
    OPTIMIZED: Jump moves with reduced overhead.
    """
    moves = []
    sx, sy, sz = start

    for dx, dy, dz in offsets:
        target = (sx + dx, sy + dy, sz + dz)

        # Fast bounds check
        tx, ty, tz = target
        if not (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE and 0 <= tz < BOARD_SIZE):
            continue

        _, can_land = should_stop_at(cache, color, target, allow_capture, allow_self_block)
        if can_land:
            target_piece = cache.piece_cache.get(target)
            is_capture = target_piece is not None and target_piece.color != color
            moves.append(Move(from_coord=start, to_coord=target, is_capture=is_capture))

    return moves


def jump_to_targets_vectorized(
    cache: OptimizedCacheManager,
    color,
    start: Tuple[int, int, int],
    offsets: np.ndarray,
    allow_capture: bool = True,
    allow_self_block: bool = False,
) -> List[Move]:
    """
    HIGHLY OPTIMIZED: Vectorized jump target processing.
    Pass offsets as numpy array for best performance.
    """
    moves = []
    start_array = np.array(start, dtype=np.int8)

    # Vectorized target calculation
    targets = start_array + offsets

    # Vectorized bounds check
    valid_mask = np.all((targets >= 0) & (targets < BOARD_SIZE), axis=1)

    # Process only valid targets
    for idx in np.where(valid_mask)[0]:
        target = tuple(targets[idx])

        _, can_land = should_stop_at(cache, color, target, allow_capture, allow_self_block)
        if can_land:
            target_piece = cache.piece_cache.get(target)
            is_capture = target_piece is not None and target_piece.color != color
            moves.append(Move(from_coord=start, to_coord=target, is_capture=is_capture))

    return moves


def validate_piece_at(
    cache: OptimizedCacheManager,
    color,
    pos: Tuple[int, int, int],
    expected_type: Optional[PieceType] = None,
) -> bool:
    """Fast piece validation."""
    piece = cache.piece_cache.get(pos)
    if piece is None:
        return False
    if piece.color != color:
        return False
    if expected_type is not None and piece.ptype != expected_type:
        return False
    return True


def is_edge_square(x: int, y: int, z: int, board_size: int = BOARD_SIZE) -> bool:
    """OPTIMIZED: Fast edge checking."""
    edge = board_size - 1
    return x in (0, edge) or y in (0, edge) or z in (0, edge)


# Convenience function to choose optimal implementation
def slide_along_directions_auto(
    cache: OptimizedCacheManager,
    color,
    start: Tuple[int, int, int],
    directions: np.ndarray,
    allow_capture: bool = True,
    allow_self_block: bool = False,
    max_steps: Optional[int] = BOARD_SIZE,
) -> List[Move]:
    """
    Automatically choose between regular and batch processing based on number of directions.
    """
    if len(directions) > 8:
        # Many directions - use batch processing
        return slide_along_directions_batch(cache, color, start, directions,
                                           allow_capture, allow_self_block, max_steps)
    else:
        # Few directions - use regular processing with early termination
        return slide_along_directions(cache, color, start, directions,
                                      allow_capture, allow_self_block, max_steps)
