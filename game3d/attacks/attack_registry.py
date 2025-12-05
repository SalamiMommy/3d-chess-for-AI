"""
Optimized Attack Registry - Precomputed attack lookup tables for fast attack detection.

Loads precomputed move/ray data and provides Numba-compatible attack checking.
"""
from __future__ import annotations
import numpy as np
import os
from numba import njit, prange
from typing import Dict, Tuple, Optional

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, SIZE, SIZE_SQUARED, VOLUME, PieceType,
    PIECE_TYPE_DTYPE, COLOR_DTYPE
)

# =============================================================================
# PRECOMPUTED DATA LOADING
# =============================================================================

_PRECOMPUTED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'movement', 'precomputed')

# Cache for loaded precomputed data
_PRECOMPUTED_MOVES_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_PRECOMPUTED_RAYS_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

def _load_precomputed_moves(piece_name: str, buffed: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load precomputed moves for a piece type.
    
    Returns: (moves_flat, offsets) or None if not available
    """
    suffix = 'buffed' if buffed else 'unbuffed'
    cache_key = f"{piece_name}_{suffix}"
    
    if cache_key in _PRECOMPUTED_MOVES_CACHE:
        return _PRECOMPUTED_MOVES_CACHE[cache_key]
    
    flat_path = os.path.join(_PRECOMPUTED_DIR, f'moves_{piece_name}_{suffix}_flat.npy')
    offsets_path = os.path.join(_PRECOMPUTED_DIR, f'moves_{piece_name}_{suffix}_offsets.npy')
    
    if not os.path.exists(flat_path) or not os.path.exists(offsets_path):
        return None
    
    try:
        moves_flat = np.load(flat_path).astype(COORD_DTYPE)
        offsets = np.load(offsets_path).astype(np.int32)
        _PRECOMPUTED_MOVES_CACHE[cache_key] = (moves_flat, offsets)
        return moves_flat, offsets
    except Exception:
        return None

def _load_precomputed_rays(piece_name: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load precomputed rays for a slider piece type.
    
    Returns: (rays_flat, ray_offsets, sq_offsets) or None if not available
    """
    if piece_name in _PRECOMPUTED_RAYS_CACHE:
        return _PRECOMPUTED_RAYS_CACHE[piece_name]
    
    flat_path = os.path.join(_PRECOMPUTED_DIR, f'rays_{piece_name}_flat.npy')
    ray_offsets_path = os.path.join(_PRECOMPUTED_DIR, f'rays_{piece_name}_ray_offsets.npy')
    sq_offsets_path = os.path.join(_PRECOMPUTED_DIR, f'rays_{piece_name}_sq_offsets.npy')
    
    if not os.path.exists(flat_path):
        return None
    
    try:
        rays_flat = np.load(flat_path).astype(COORD_DTYPE)
        ray_offsets = np.load(ray_offsets_path).astype(np.int32)
        sq_offsets = np.load(sq_offsets_path).astype(np.int32)
        _PRECOMPUTED_RAYS_CACHE[piece_name] = (rays_flat, ray_offsets, sq_offsets)
        return rays_flat, ray_offsets, sq_offsets
    except Exception:
        return None


# =============================================================================
# PRELOAD ALL ATTACK DATA AT MODULE LOAD
# =============================================================================

# Piece types that use jump-based attacks (offset check)
JUMP_PIECE_TYPES = [
    'KNIGHT', 'KNIGHT31', 'KNIGHT32', 'KING', 'PRIEST', 'SPEEDER', 'SLOWER',
    'FREEZER', 'HIVE', 'ARMOUR', 'ARCHER', 'BOMB', 'SWAPPER', 'PANEL',
    'GEOMANCER', 'MIRROR', 'INFILTRATOR', 'BLACKHOLE', 'WHITEHOLE', 'ECHO',
    'NEBULA', 'ORBITER'
]

# Piece types that use slider-based attacks (raycast check)
SLIDER_PIECE_TYPES = [
    'BISHOP', 'ROOK', 'QUEEN', 'TRIGONALBISHOP', 'SPIRAL', 
    'XYQUEEN', 'XZQUEEN', 'YZQUEEN', 'VECTORSLIDER', 'CONESLIDER',
    'TRAILBLAZER', 'REFLECTOR'
]

# Build mapping from PieceType enum value to index for array-based lookup
_PTYPE_TO_IDX = {}
_IDX_TO_PTYPE = {}
_ATTACK_OFFSETS_BY_IDX = {}  # idx -> (offsets_flat, sq_offsets)

def _initialize_attack_registry():
    """Initialize attack lookup tables from precomputed data."""
    global _PTYPE_TO_IDX, _IDX_TO_PTYPE, _ATTACK_OFFSETS_BY_IDX
    
    idx = 0
    for piece_name in JUMP_PIECE_TYPES:
        try:
            ptype = getattr(PieceType, piece_name)
            data = _load_precomputed_moves(piece_name, buffed=False)
            if data is not None:
                _PTYPE_TO_IDX[ptype.value] = idx
                _IDX_TO_PTYPE[idx] = ptype.value
                _ATTACK_OFFSETS_BY_IDX[idx] = data
                idx += 1
        except AttributeError:
            continue  # Piece type doesn't exist

# Initialize at module load
_initialize_attack_registry()


# =============================================================================
# NUMBA ATTACK KERNELS
# =============================================================================

@njit(cache=True, fastmath=True)
def _coord_to_flat_idx(x: int, y: int, z: int) -> int:
    """Convert coordinate to flat index for precomputed lookup."""
    return x + y * SIZE + z * SIZE_SQUARED

@njit(cache=True, fastmath=True)
def _check_jump_attack_precomputed(
    target: np.ndarray,
    attacker: np.ndarray,
    moves_flat: np.ndarray,
    sq_offsets: np.ndarray
) -> bool:
    """Check if a jump piece at 'attacker' can attack 'target' using precomputed offsets.
    
    Uses O(n) scan through precomputed moves for the attacker's square.
    """
    ax, ay, az = attacker[0], attacker[1], attacker[2]
    tx, ty, tz = target[0], target[1], target[2]
    
    # Get flat index for attacker position
    flat_idx = _coord_to_flat_idx(ax, ay, az)
    
    if flat_idx < 0 or flat_idx >= len(sq_offsets) - 1:
        return False
    
    # Get range of moves for this square
    start = sq_offsets[flat_idx]
    end = sq_offsets[flat_idx + 1]
    
    # Check each precomputed move destination
    for i in range(start, end):
        mx, my, mz = moves_flat[i, 0], moves_flat[i, 1], moves_flat[i, 2]
        if mx == tx and my == ty and mz == tz:
            return True
    
    return False

@njit(cache=True, fastmath=True)
def _check_slider_attack_precomputed(
    target: np.ndarray,
    attacker: np.ndarray,
    occ: np.ndarray,
    rays_flat: np.ndarray,
    ray_offsets: np.ndarray,
    sq_offsets: np.ndarray
) -> bool:
    """Check if a slider at 'attacker' can attack 'target' using precomputed rays.
    
    Traces each ray from attacker until blocked or reaches target.
    """
    ax, ay, az = attacker[0], attacker[1], attacker[2]
    tx, ty, tz = target[0], target[1], target[2]
    
    # Get flat index for attacker position
    flat_idx = _coord_to_flat_idx(ax, ay, az)
    
    if flat_idx < 0 or flat_idx >= len(sq_offsets) - 1:
        return False
    
    # Get range of rays for this square
    ray_start = sq_offsets[flat_idx]
    ray_end = sq_offsets[flat_idx + 1]
    
    # Check each ray
    for r in range(ray_start, ray_end):
        # Get squares along this ray
        sq_start = ray_offsets[r]
        sq_end = ray_offsets[r + 1]
        
        # Trace along ray
        for s in range(sq_start, sq_end):
            rx, ry, rz = rays_flat[s, 0], rays_flat[s, 1], rays_flat[s, 2]
            
            # Check if we hit target
            if rx == tx and ry == ty and rz == tz:
                return True
            
            # Check if blocked
            if occ[rx, ry, rz] != 0:
                break  # Ray is blocked, try next ray
    
    return False


@njit(cache=True, fastmath=True, parallel=False)  # parallel=False for cache efficiency
def _fast_attack_kernel_extended(
    target: np.ndarray,
    attacker_coords: np.ndarray,
    attacker_types: np.ndarray,
    occ: np.ndarray,
    attacker_color: int,
    # Precomputed data for sliders
    bishop_rays: np.ndarray, bishop_ray_offsets: np.ndarray, bishop_sq_offsets: np.ndarray,
    rook_rays: np.ndarray, rook_ray_offsets: np.ndarray, rook_sq_offsets: np.ndarray,
    queen_rays: np.ndarray, queen_ray_offsets: np.ndarray, queen_sq_offsets: np.ndarray,
    trigonal_rays: np.ndarray, trigonal_ray_offsets: np.ndarray, trigonal_sq_offsets: np.ndarray,
    spiral_rays: np.ndarray, spiral_ray_offsets: np.ndarray, spiral_sq_offsets: np.ndarray,
    skipped_indices: np.ndarray
) -> int:
    """
    Extended attack kernel with more inline piece handling.
    
    Returns:
        0: Not attacked (by checked pieces)
        1: Attacked
    
    Populates skipped_indices with indices of pieces that were not checked.
    """
    n = attacker_coords.shape[0]
    tx, ty, tz = target[0], target[1], target[2]
    
    skipped_count = 0
    
    for i in range(n):
        ax, ay, az = attacker_coords[i, 0], attacker_coords[i, 1], attacker_coords[i, 2]
        atype = attacker_types[i]
        
        is_attacking = False
        handled = True
        
        # === PAWN (type 1) ===
        if atype == 1:
            dz = tz - az
            if attacker_color == 1:  # White
                if dz == 1 and abs(tx - ax) == 1 and abs(ty - ay) == 1:
                    is_attacking = True
            else:  # Black
                if dz == -1 and abs(tx - ax) == 1 and abs(ty - ay) == 1:
                    is_attacking = True
                    
        # === KNIGHT (type 2) ===
        elif atype == 2:
            dx = abs(tx - ax)
            dy = abs(ty - ay)
            dz = abs(tz - az)
            if (dx == 1 and dy == 2 and dz == 0) or \
               (dx == 2 and dy == 1 and dz == 0) or \
               (dx == 1 and dy == 0 and dz == 2) or \
               (dx == 0 and dy == 1 and dz == 2) or \
               (dx == 2 and dy == 0 and dz == 1) or \
               (dx == 0 and dy == 2 and dz == 1):
                is_attacking = True
                
        # === BISHOP (type 3) ===
        elif atype == 3:
            if bishop_rays.size > 0:
                is_attacking = _check_slider_attack_precomputed(
                    target, attacker_coords[i], occ,
                    bishop_rays, bishop_ray_offsets, bishop_sq_offsets
                )
            else:
                handled = False
                
        # === ROOK (type 4) ===
        elif atype == 4:
            if rook_rays.size > 0:
                is_attacking = _check_slider_attack_precomputed(
                    target, attacker_coords[i], occ,
                    rook_rays, rook_ray_offsets, rook_sq_offsets
                )
            else:
                handled = False
                
        # === QUEEN (type 5) ===
        elif atype == 5:
            if queen_rays.size > 0:
                is_attacking = _check_slider_attack_precomputed(
                    target, attacker_coords[i], occ,
                    queen_rays, queen_ray_offsets, queen_sq_offsets
                )
            else:
                handled = False
                
        # === KING (type 6) or PRIEST (type 7) ===
        elif atype == 6 or atype == 7:
            dx = abs(tx - ax)
            dy = abs(ty - ay)
            dz = abs(tz - az)
            if dx <= 1 and dy <= 1 and dz <= 1 and (dx + dy + dz > 0):
                is_attacking = True
                
        # === TRIGONAL BISHOP (type 10) ===
        elif atype == 10:
            if trigonal_rays.size > 0:
                is_attacking = _check_slider_attack_precomputed(
                    target, attacker_coords[i], occ,
                    trigonal_rays, trigonal_ray_offsets, trigonal_sq_offsets
                )
            else:
                handled = False
        
        # === KNIGHT31 (type 8) ===
        elif atype == 8:
            dx = abs(tx - ax)
            dy = abs(ty - ay)
            dz = abs(tz - az)
            # (3, 1, 0) permutations
            if (dx == 3 and dy == 1 and dz == 0) or \
               (dx == 1 and dy == 3 and dz == 0) or \
               (dx == 3 and dy == 0 and dz == 1) or \
               (dx == 0 and dy == 3 and dz == 1) or \
               (dx == 1 and dy == 0 and dz == 3) or \
               (dx == 0 and dy == 1 and dz == 3):
                is_attacking = True
                
        # === KNIGHT32 (type 9) ===
        elif atype == 9:
            dx = abs(tx - ax)
            dy = abs(ty - ay)
            dz = abs(tz - az)
            # (3, 2, 0) permutations
            if (dx == 3 and dy == 2 and dz == 0) or \
               (dx == 2 and dy == 3 and dz == 0) or \
               (dx == 3 and dy == 0 and dz == 2) or \
               (dx == 0 and dy == 3 and dz == 2) or \
               (dx == 2 and dy == 0 and dz == 3) or \
               (dx == 0 and dy == 2 and dz == 3):
                is_attacking = True
                
        # === SPIRAL (type 38) ===
        elif atype == 38:
            if spiral_rays.size > 0:
                is_attacking = _check_slider_attack_precomputed(
                    target, attacker_coords[i], occ,
                    spiral_rays, spiral_ray_offsets, spiral_sq_offsets
                )
            else:
                handled = False
        
        # === SPEEDER, SLOWER, FREEZER, HIVE, ARMOUR, SWAPPER (king-like movers) ===
        elif atype in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26):
            # These are king-like movers (1 square in any direction)
            dx = abs(tx - ax)
            dy = abs(ty - ay)
            dz = abs(tz - az)
            if dx <= 1 and dy <= 1 and dz <= 1 and (dx + dy + dz > 0):
                is_attacking = True
        
        # === PANEL (type 37) - special 2D movement ===
        elif atype == 37:
            dx = tx - ax
            dy = ty - ay
            dz = tz - az
            # Panel moves in X-Y plane only (dz=0), up to 2 squares
            if dz == 0 and abs(dx) <= 2 and abs(dy) <= 2 and (abs(dx) + abs(dy) > 0):
                is_attacking = True
        
        else:
            handled = False
            skipped_indices[skipped_count + 1] = i
            skipped_count += 1
            
        if handled and is_attacking:
            return 1  # Attacked!
            
    skipped_indices[0] = skipped_count
    return 0


# =============================================================================
# PUBLIC API
# =============================================================================

# Preload slider ray data at module load for fast access
_BISHOP_RAYS = _load_precomputed_rays('BISHOP')
_ROOK_RAYS = _load_precomputed_rays('ROOK')
_QUEEN_RAYS = _load_precomputed_rays('QUEEN')
_TRIGONAL_RAYS = _load_precomputed_rays('TRIGONALBISHOP')
_SPIRAL_RAYS = _load_precomputed_rays('SPIRAL')

# Create empty arrays for fallback
_EMPTY_RAYS = np.empty((0, 3), dtype=COORD_DTYPE)
_EMPTY_OFFSETS = np.zeros(1, dtype=np.int32)

def get_slider_ray_data(piece_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get precomputed ray data for a slider piece type."""
    data = _load_precomputed_rays(piece_name)
    if data is None:
        return _EMPTY_RAYS, _EMPTY_OFFSETS, _EMPTY_OFFSETS
    return data

def square_attacked_by_extended(
    board,
    square: np.ndarray,
    attacker_color: int,
    cache
) -> bool:
    """
    Optimized attack detection using extended Numba kernel with precomputed data.
    
    This replaces the original square_attacked_by_fast with more inline piece handling.
    """
    if cache is None or not hasattr(cache, 'occupancy_cache'):
        from game3d.attacks.check import _square_attacked_by_slow
        return _square_attacked_by_slow(board, square, attacker_color, cache)
        
    # Get all attackers
    attacker_positions = cache.occupancy_cache.get_positions(attacker_color)
    if attacker_positions.shape[0] == 0:
        return False
        
    # Get types
    _, attacker_types = cache.occupancy_cache.batch_get_attributes_unsafe(attacker_positions)
    
    # Prepare skipped array
    skipped_indices = np.zeros(attacker_positions.shape[0] + 1, dtype=np.int32)
    
    # Get precomputed ray data
    bishop_data = _BISHOP_RAYS if _BISHOP_RAYS else (_EMPTY_RAYS, _EMPTY_OFFSETS, _EMPTY_OFFSETS)
    rook_data = _ROOK_RAYS if _ROOK_RAYS else (_EMPTY_RAYS, _EMPTY_OFFSETS, _EMPTY_OFFSETS)
    queen_data = _QUEEN_RAYS if _QUEEN_RAYS else (_EMPTY_RAYS, _EMPTY_OFFSETS, _EMPTY_OFFSETS)
    trigonal_data = _TRIGONAL_RAYS if _TRIGONAL_RAYS else (_EMPTY_RAYS, _EMPTY_OFFSETS, _EMPTY_OFFSETS)
    spiral_data = _SPIRAL_RAYS if _SPIRAL_RAYS else (_EMPTY_RAYS, _EMPTY_OFFSETS, _EMPTY_OFFSETS)
    
    # Run extended kernel
    result = _fast_attack_kernel_extended(
        square.astype(COORD_DTYPE),
        attacker_positions.astype(COORD_DTYPE),
        attacker_types.astype(PIECE_TYPE_DTYPE),
        cache.occupancy_cache._occ,
        int(attacker_color),
        bishop_data[0], bishop_data[1], bishop_data[2],
        rook_data[0], rook_data[1], rook_data[2],
        queen_data[0], queen_data[1], queen_data[2],
        trigonal_data[0], trigonal_data[1], trigonal_data[2],
        spiral_data[0], spiral_data[1], spiral_data[2],
        skipped_indices
    )
    
    if result == 1:
        return True
        
    # Handle any remaining skipped pieces with fallback
    skipped_count = int(skipped_indices[0])
    if skipped_count == 0:
        return False
        
    # Fallback for unhandled piece types
    from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch
    from game3d.game.gamestate import GameState
    
    indices = skipped_indices[1:skipped_count+1]
    unhandled_positions = attacker_positions[indices]
    
    dummy_state = GameState(board, attacker_color, cache)
    moves = generate_pseudolegal_moves_batch(dummy_state, unhandled_positions)
    
    if moves.size == 0:
        return False
        
    target_x, target_y, target_z = square[0], square[1], square[2]
    hits = (moves[:, 3] == target_x) & (moves[:, 4] == target_y) & (moves[:, 5] == target_z)
    
    return bool(np.any(hits))


__all__ = [
    'square_attacked_by_extended',
    'get_slider_ray_data',
    '_load_precomputed_moves',
    '_load_precomputed_rays',
]
