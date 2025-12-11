"""
Optimized Attack Registry - Precomputed attack lookup tables for fast attack detection.

Loads precomputed move/ray data and provides Numba-compatible attack checking via generic lookup tables.
"""
from __future__ import annotations
import numpy as np
import os
from numba import njit, prange
from typing import Dict, Tuple, Optional, List

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, SIZE, SIZE_SQUARED, VOLUME, PieceType,
    PIECE_TYPE_DTYPE, COLOR_DTYPE, MAX_HISTORY_SIZE, HASH_DTYPE, INDEX_DTYPE
)
# Deferred imports to avoid circular deps if possible, but for type hint/constants usually fine
from game3d.core.buffer import GameBuffer
from game3d.core.api import generate_pseudolegal_moves

# =============================================================================
# PRECOMPUTED DATA LOADING
# =============================================================================

_PRECOMPUTED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'movement', 'precomputed')

# Constants for array sizes
_MAX_PIECE_TYPE = 64
_SIZE_CUBED_PLUS_ONE = VOLUME + 1

def _load_numpy_safe(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None

def _initialize_registries():
    """
    Initialize generic lookup tables for Jump and Slider pieces.
    Returns:
        jump_data: (moves_flat, sq_offsets, type_map)
        slider_data: (rays_flat, ray_offsets, sq_offsets, type_map)
    """
    
    # --- 1. JUMP REGISTRY ---
    jump_moves_list = []
    jump_sq_offsets_list = []
    
    # type_map: [sq_offset_start_idx, unused] -> we just need start idx in the concatenated sq_offsets
    # But wait, sq_offsets is fixed size (VOLUME+1) for each piece.
    # So we can just flatten jump_sq_offsets_list and index by (type_idx * (VOLUME+1)).
    # BUT, the VALUES in sq_offsets need to be adjusted by cumulative moves count.
    
    jump_type_map = np.full((_MAX_PIECE_TYPE, 2), -1, dtype=np.int32)
    
    current_move_offset = 0
    current_sq_offset_start = 0
    
    # We iterate all PieceTypes. 
    # If a file moves_{NAME}_unbuffed_flat.npy exists, we load it.
    
    # Standard Pieces that use Jump logic (optionally): Knight, King.
    # We treat them as generic jump pieces if loaded here, 
    # but the kernel might handle them explicitly for speed.
    
    valid_jump_types = []
    
    for pt in PieceType:
        if pt.value <= 0: continue
        name = pt.name
        
        # Try loading UNBUFFED moves (Standard attack check uses unbuffed logic mostly? 
        # Actually buffs might affect attacks (e.g. Speeder buffing Range). 
        # But 'captured' pieces don't usually get buffs from the capturer? 
        # Attack check: "Can piece at X attack Y?" 
        # Usually we assume standard movement unless context provided.
        # For simple check, extended kernel currently uses unbuffed usually?
        # The previous code loaded 'moves_KNIGHT_unbuffed' inside JUMP_PIECE_TYPES loop?
        # It used `_load_precomputed_moves(..., buffed=False)`.
        # So we stick to unbuffed for "base attacks".
        
        flat_path = os.path.join(_PRECOMPUTED_DIR, f'moves_{name}_unbuffed_flat.npy')
        off_path = os.path.join(_PRECOMPUTED_DIR, f'moves_{name}_unbuffed_offsets.npy')
        
        moves = _load_numpy_safe(flat_path)
        offsets = _load_numpy_safe(off_path)
        
        if moves is not None and offsets is not None:
             # Ensure types
            moves = moves.astype(COORD_DTYPE)
            offsets = offsets.astype(np.int32)
            
            # Adjust offsets
            offsets_adjusted = offsets + current_move_offset
            
            jump_moves_list.append(moves)
            jump_sq_offsets_list.append(offsets_adjusted)
            
            # Record map
            # We store the index into the LIST of pieces? 
            # Or the index into the FLATTENED sq_offsets array?
            # Index into flattened sq_offsets array sounds best.
            jump_type_map[pt.value, 0] = current_sq_offset_start
            jump_type_map[pt.value, 1] = 1 # Valid
            
            current_move_offset += len(moves)
            current_sq_offset_start += len(offsets)
            valid_jump_types.append(pt.value)

    if jump_moves_list:
        jump_moves_flat = np.concatenate(jump_moves_list)
        jump_sq_offsets_flat = np.concatenate(jump_sq_offsets_list)
    else:
        jump_moves_flat = np.zeros((0, 3), dtype=COORD_DTYPE)
        jump_sq_offsets_flat = np.zeros((0,), dtype=np.int32)

    # --- 2. SLIDER REGISTRY ---
    slider_rays_list = []
    slider_ray_offsets_list = []
    slider_sq_offsets_list = []
    
    slider_type_map = np.full((_MAX_PIECE_TYPE, 2), -1, dtype=np.int32)
    
    current_ray_idx_offset = 0 # Offset for values in ray_offsets (indices into rays)
    current_ray_offset_arr_offset = 0 # Offset for values in sq_offsets (indices into ray_offsets)
    
    current_sq_offset_start_idx = 0 # Index in flattened sq_offsets where this piece starts
    
    # Explicitly check slider files
    # rays_{NAME}_flat.npy
    
    for pt in PieceType:
        if pt.value <= 0: continue
        name = pt.name
        
        flat_path = os.path.join(_PRECOMPUTED_DIR, f'rays_{name}_flat.npy')
        ray_off_path = os.path.join(_PRECOMPUTED_DIR, f'rays_{name}_ray_offsets.npy')
        sq_off_path = os.path.join(_PRECOMPUTED_DIR, f'rays_{name}_sq_offsets.npy')
        
        rays = _load_numpy_safe(flat_path)
        ray_offs = _load_numpy_safe(ray_off_path)
        sq_offs = _load_numpy_safe(sq_off_path)
        
        if rays is not None and ray_offs is not None and sq_offs is not None:
            rays = rays.astype(COORD_DTYPE)
            ray_offs = ray_offs.astype(np.int32)
            sq_offs = sq_offs.astype(np.int32)
            
            # Adjust ray_offsets values (they point to rays)
            ray_offs_adjusted = ray_offs + current_ray_idx_offset
            
            # Adjust sq_offsets values (they point to ray_offsets)
            sq_offs_adjusted = sq_offs + current_ray_offset_arr_offset
            
            slider_rays_list.append(rays)
            slider_ray_offsets_list.append(ray_offs_adjusted)
            slider_sq_offsets_list.append(sq_offs_adjusted)
            
            slider_type_map[pt.value, 0] = current_sq_offset_start_idx
            slider_type_map[pt.value, 1] = 1 # Valid
            
            current_ray_idx_offset += len(rays)
            current_ray_offset_arr_offset += len(ray_offs)
            current_sq_offset_start_idx += len(sq_offs)

    if slider_rays_list:
        slider_rays_flat = np.concatenate(slider_rays_list)
        slider_ray_offsets_flat = np.concatenate(slider_ray_offsets_list)
        slider_sq_offsets_flat = np.concatenate(slider_sq_offsets_list)
    else:
        slider_rays_flat = np.zeros((0, 3), dtype=COORD_DTYPE)
        slider_ray_offsets_flat = np.zeros((0,), dtype=np.int32)
        slider_sq_offsets_flat = np.zeros((0,), dtype=np.int32)
        
    return (
        jump_moves_flat, jump_sq_offsets_flat, jump_type_map,
        slider_rays_flat, slider_ray_offsets_flat, slider_sq_offsets_flat, slider_type_map
    )

# Load data at module level
(
    _JUMP_MOVES_FLAT, _JUMP_SQ_OFFSETS, _JUMP_TYPE_MAP,
    _SLIDER_RAYS_FLAT, _SLIDER_RAY_OFFSETS, _SLIDER_SQ_OFFSETS, _SLIDER_TYPE_MAP
) = _initialize_registries()


# =============================================================================
# NUMBA ATTACK KERNELS
# =============================================================================

@njit(cache=True, fastmath=True)
def _coord_to_flat_idx(x: int, y: int, z: int) -> int:
    """Convert coordinate to flat index for precomputed lookup."""
    return x + y * SIZE + z * SIZE_SQUARED

@njit(cache=True, fastmath=True)
def _check_generic_jump_attack(
    target: np.ndarray,
    attacker: np.ndarray,
    moves_flat: np.ndarray,
    sq_offsets: np.ndarray,
    offset_start_idx: int
) -> bool:
    """Check jump attack using generic concatenated tables."""
    ax, ay, az = attacker[0], attacker[1], attacker[2]
    tx, ty, tz = target[0], target[1], target[2]
    
    flat_idx = _coord_to_flat_idx(ax, ay, az)
    
    # Locate range in sq_offsets
    # sq_offsets contains [start, end] for each square.
    # The segment for this piece starts at offset_start_idx.
    
    base_idx = offset_start_idx + flat_idx
    if base_idx >= len(sq_offsets) - 1: return False # Bounds check
    
    move_start = sq_offsets[base_idx]
    move_end = sq_offsets[base_idx + 1]
    
    for i in range(move_start, move_end):
        mx, my, mz = moves_flat[i, 0], moves_flat[i, 1], moves_flat[i, 2]
        if mx == tx and my == ty and mz == tz:
            return True
            
    return False

@njit(cache=True, fastmath=True)
def _check_generic_slider_attack(
    target: np.ndarray,
    attacker: np.ndarray,
    occ: np.ndarray,
    rays_flat: np.ndarray,
    ray_offsets: np.ndarray,
    sq_offsets: np.ndarray,
    offset_start_idx: int
) -> bool:
    """Check slider attack using generic concatenated tables."""
    ax, ay, az = attacker[0], attacker[1], attacker[2]
    tx, ty, tz = target[0], target[1], target[2]
    
    flat_idx = _coord_to_flat_idx(ax, ay, az)
    
    base_idx = offset_start_idx + flat_idx
    if base_idx >= len(sq_offsets) - 1: return False
    
    ray_start = sq_offsets[base_idx]
    ray_end = sq_offsets[base_idx + 1]
    
    for r in range(ray_start, ray_end):
        # Ray range in rays_flat
        r_start = ray_offsets[r]
        r_end = ray_offsets[r+1]
        
        for s in range(r_start, r_end):
            rx, ry, rz = rays_flat[s, 0], rays_flat[s, 1], rays_flat[s, 2]
            
            if rx == tx and ry == ty and rz == tz:
                return True
            
            if occ[rx, ry, rz] != 0:
                break
                
    return False

@njit(cache=True, fastmath=True, parallel=False)
def _fast_attack_kernel_extended(
    target: np.ndarray,
    attacker_coords: np.ndarray,
    ptype_grid: np.ndarray,  # ✅ OPTIMIZED: pass full grid
    occ: np.ndarray,
    attacker_color: int,
    # Generic Jump Tables
    jump_moves: np.ndarray, jump_sq_offsets: np.ndarray, jump_map: np.ndarray,
    # Generic Slider Tables
    slider_rays: np.ndarray, slider_ray_offsets: np.ndarray, slider_sq_offsets: np.ndarray, slider_map: np.ndarray,
    # Output
    skipped_indices: np.ndarray
) -> int:
    """
    Consolidated attack kernel handling all pieces via generic lookup or inline logic.
    """
    n = attacker_coords.shape[0]
    tx, ty, tz = target[0], target[1], target[2]
    
    skipped_count = 0
    
    for i in range(n):
        ax, ay, az = attacker_coords[i, 0], attacker_coords[i, 1], attacker_coords[i, 2]
        atype = ptype_grid[ax, ay, az]  # ✅ Direct lookup
        
        is_attacking = False
        handled = True
        
        # --- 1. Common Types Inline (for speed) ---
        
        if atype == 1: # PAWN
            dz = tz - az
            if attacker_color == 1:  # White
                if dz == 1 and abs(tx - ax) == 1 and abs(ty - ay) == 1:
                    is_attacking = True
            else:  # Black
                if dz == -1 and abs(tx - ax) == 1 and abs(ty - ay) == 1:
                    is_attacking = True
                    
        elif atype == 6 or atype == 7: # KING, PRIEST (Simple Range 1 adjacent)
            dx = abs(tx - ax)
            dy = abs(ty - ay)
            dz = abs(tz - az)
            if dx <= 1 and dy <= 1 and dz <= 1 and (dx + dy + dz > 0):
                is_attacking = True
                
        # --- 2. Generic Lookup ---
        else:
            # Try Slider Registry
            if atype < 64:
                # Check Slider
                s_idx = slider_map[atype, 0]
                is_valid_s = slider_map[atype, 1]
                
                if is_valid_s == 1:
                    is_attacking = _check_generic_slider_attack(
                        target, attacker_coords[i], occ,
                        slider_rays, slider_ray_offsets, slider_sq_offsets, s_idx
                    )
                else:
                    # Check Jump Registry
                    j_idx = jump_map[atype, 0]
                    is_valid_j = jump_map[atype, 1]
                    
                    if is_valid_j == 1:
                        is_attacking = _check_generic_jump_attack(
                            target, attacker_coords[i], 
                            jump_moves, jump_sq_offsets, j_idx
                        )
                    else:
                        handled = False
            else:
                handled = False

        if not handled:
            skipped_indices[skipped_count + 1] = i
            skipped_count += 1
        elif is_attacking:
            return 1
            
    skipped_indices[0] = skipped_count
    return 0


# =============================================================================
# PUBLIC API
# =============================================================================

def square_attacked_by_extended(
    board,
    square: np.ndarray,
    attacker_color: int,
    cache
) -> bool:
    """
    Optimized attack detection using generic precomputed data.
    """
    if cache is None or not hasattr(cache, 'occupancy_cache'):
        from game3d.attacks.check import _square_attacked_by_slow
        return _square_attacked_by_slow(board, square, attacker_color, cache)
        
    # Get all attackers
    attacker_positions = cache.occupancy_cache.get_positions(attacker_color)
    if attacker_positions.shape[0] == 0:
        return False
        
    # ✅ OPTIMIZED: Direct grid access, no intermediate allocation/lookup
    # _, attacker_types = cache.occupancy_cache.batch_get_attributes_unsafe(attacker_positions)
    


    # Prepare skipped array
    # ✅ OPTIMIZED: Use empty (faster than zeros) + manual init of counter
    skipped_indices = np.empty(attacker_positions.shape[0] + 1, dtype=np.int32)
    skipped_indices[0] = 0
    
    # Run generic kernel
    result = _fast_attack_kernel_extended(
        square.astype(COORD_DTYPE),
        attacker_positions.astype(COORD_DTYPE),
        cache.occupancy_cache._ptype, # ✅ Pass full grid
        cache.occupancy_cache._occ,
        int(attacker_color),
        _JUMP_MOVES_FLAT, _JUMP_SQ_OFFSETS, _JUMP_TYPE_MAP,
        _SLIDER_RAYS_FLAT, _SLIDER_RAY_OFFSETS, _SLIDER_SQ_OFFSETS, _SLIDER_TYPE_MAP,
        skipped_indices
    )
    
    if result == 1:
        return True
        
    # Handle skipped (Should be very rare now)
    skipped_count = int(skipped_indices[0])
    # print(f"DEBUG: Result {result}, Skipped {skipped_count}")
    if skipped_count == 0:
        return False
        
    # Fallback for truly unhandled piece types
    
    indices = skipped_indices[1:skipped_count+1]
    unhandled_positions = attacker_positions[indices]
    
    # We need types for the fallback buffer construction
    # Since this is the fallback path (rare), we can afford the lookup now
    # Or just use the ptype grid we have access to?
    # batch_get_attributes_unsafe is fast enough for small N
    
    # For correctness in buffer construction, we need the types
    # Let's just fetch them for the specific pieces
    _, unhandled_types = cache.occupancy_cache.batch_get_attributes_unsafe(unhandled_positions)
    
    # ✅ OPTIMIZATION: Construct a minimal GameBuffer with ONLY unhandled pieces
    # This forces generate_pseudolegal_moves to only process these pieces, 
    # avoiding generating moves for the entire board (30+ pieces) when only 1 is unhandled.
    
    occ_grid = cache.occupancy_cache._occ
    ptype_grid = cache.occupancy_cache._ptype
    # We still need a flat view for some kernels
    board_color_flat = occ_grid.flatten(order='F')
    
    n_unhandled = unhandled_positions.shape[0]
    max_pieces = max(512, n_unhandled)
    
    # Pre-allocate sparse arrays
    occupied_coords = np.zeros((max_pieces, 3), dtype=COORD_DTYPE)
    occupied_types = np.zeros(max_pieces, dtype=PIECE_TYPE_DTYPE)
    occupied_colors = np.zeros(max_pieces, dtype=COLOR_DTYPE)
    
    if n_unhandled > 0:
        occupied_coords[:n_unhandled] = unhandled_positions
        occupied_types[:n_unhandled] = unhandled_types
        # Start color is 1 or 2. attacker_color is likely 1 or 2.
        occupied_colors[:n_unhandled] = attacker_color
    
    # Use dummy meta (only active color matters for generation)
    meta = np.zeros(10, dtype=INDEX_DTYPE)
    meta[0] = attacker_color
    
    # Use dummy history
    history = np.zeros(MAX_HISTORY_SIZE, dtype=HASH_DTYPE)
    
    # Empty aura maps (not relevant for basic attack check)
    is_buffed = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
    is_debuffed = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
    is_frozen = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
    
    # Create minimal buffer without copying dense arrays
    buffer = GameBuffer(
        occupied_coords,
        occupied_types,
        occupied_colors,
        n_unhandled,     # Only unhandled pieces are active!
        ptype_grid,      # Read-only access to existing grid
        occ_grid,        # Read-only access to existing grid
        board_color_flat,
        is_buffed,
        is_debuffed,
        is_frozen,
        meta,
        0, # zkey (unused)
        history,
        0 # history_count
    )
    
    # Generate moves ONLY for these pieces
    moves = generate_pseudolegal_moves(buffer)
    
    if moves.size == 0:
        return False
        
    # Check if any move hits the target square
    target_x, target_y, target_z = square[0], square[1], square[2]
    hits = (moves[:, 3] == target_x) & (moves[:, 4] == target_y) & (moves[:, 5] == target_z)
    
    return bool(np.any(hits))

__all__ = [
    'square_attacked_by_extended',
]
