
"""
Functional Move Generation.
Orchestrates move generation for GameBuffer using Numba kernels.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple

from game3d.core.buffer import GameBuffer
from game3d.common.shared_types import (
    COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE,
    SIZE, SIZE_SQUARED, SIZE_MINUS_1,
    PieceType, N_PIECE_TYPES,
    PAWN_START_RANK_WHITE, PAWN_START_RANK_BLACK,
    COLOR_WHITE, COLOR_BLACK
)

# --- 1. Import Vectors ---
from game3d.pieces.pieces.pawn import (
    PAWN_PUSH_DIRECTIONS, PAWN_ATTACK_DIRECTIONS_WHITE, PAWN_ATTACK_DIRECTIONS_BLACK
)
from game3d.pieces.pieces.knight import KNIGHT_MOVEMENT_VECTORS
from game3d.pieces.pieces.bishop import BISHOP_MOVEMENT_VECTORS
from game3d.pieces.pieces.rook import ROOK_MOVEMENT_VECTORS
from game3d.pieces.pieces.queen import QUEEN_MOVEMENT_VECTORS
from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS
from game3d.pieces.pieces.trigonalbishop import TRIGONAL_BISHOP_VECTORS
from game3d.pieces.pieces.wall import WALL_BLOCK_OFFSETS

# Extended Piece Vectors (Imported or Defined)
from game3d.pieces.pieces.bigknights import (
    KNIGHT31_MOVEMENT_VECTORS, BUFFED_KNIGHT31_MOVEMENT_VECTORS,
    KNIGHT32_MOVEMENT_VECTORS, BUFFED_KNIGHT32_MOVEMENT_VECTORS
)
from game3d.pieces.pieces.edgerook import _edgerook_bfs_batch_numba, _EDGE_NEIGHBORS, EDGE_ROOK_VECTORS
from game3d.pieces.pieces.orbiter import _ORBITAL_DIRS, _BUFFED_ORBITAL_DIRS
from game3d.pieces.pieces.vectorslider import VECTOR_DIRECTIONS
from game3d.pieces.pieces.facecone import FACE_CONE_MOVEMENT_VECTORS
from game3d.pieces.pieces.mirror import _generate_mirror_moves_kernel

# Final Set of Imports
from game3d.pieces.pieces.xzzigzag import XZ_ZIGZAG_DIRECTIONS
from game3d.pieces.pieces.yzzigzag import YZ_ZIGZAG_DIRECTIONS
from game3d.pieces.pieces.reflector import _trace_reflector_rays_batch, _REFLECTOR_DIRS
from game3d.pieces.pieces.spiral import SPIRAL_MOVEMENT_VECTORS, MAX_SPIRAL_DISTANCE
from game3d.pieces.pieces.trailblazer import ROOK_DIRECTIONS, MAX_TRAILBLAZER_DISTANCE
from game3d.pieces.pieces.infiltrator import _get_pawn_targets_kernel, _generate_infiltrator_teleport_moves

# New imports for missing pieces
from game3d.pieces.pieces.swapper import _generate_swap_moves_kernel
from game3d.pieces.pieces.friendlytp import _generate_friendlytp_moves_fused
from game3d.pieces.pieces.geomancer import GEOMANCY_OFFSETS, BUFFED_GEOMANCY_OFFSETS, _generate_geomancy_moves_kernel
from game3d.pieces.pieces.echo import _ECHO_DIRECTIONS, _BUFFED_ECHO_DIRECTIONS

# Define Plane Queen Vectors Locally (avoiding circular imports)
_XY_SLIDER_DIRS = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
], dtype=COORD_DTYPE)

_XZ_SLIDER_DIRS = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
], dtype=COORD_DTYPE)

_YZ_SLIDER_DIRS = np.array([
    [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
], dtype=COORD_DTYPE)

# King 3D dirs (26)
dx, dy, dz = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_c = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
_KING_3D_DIRS = all_c[np.any(all_c != 0, axis=1)].astype(COORD_DTYPE)


# --- 2. Import Kernels ---
from game3d.pieces.pieces.pawn import _generate_pawn_moves_batch_kernel
# We alias engine kernels for clarity
from game3d.movement.jump_engine import _generate_and_filter_jump_moves_batch
from game3d.movement.slider_engine import _generate_all_slider_moves_batch
from game3d.pieces.pieces.wall import _generate_wall_moves_fused_kernel, _filter_anchors_numba

# --- 3. Constants / Buff Helper ---


# --- 3. Constants / Aura Helper ---

# Radius 2 Offsets for Aura Broadcasting (Chebyshev distance)
# max(|dx|, |dy|, |dz|) <= 2
_r2_offsets_list = []
for x in range(-2, 3):
    for y in range(-2, 3):
        for z in range(-2, 3):
            if x == 0 and y == 0 and z == 0: continue
            if max(abs(x), abs(y), abs(z)) <= 2:  # Chebyshev distance
                _r2_offsets_list.append([x, y, z])
RADIUS_2_OFFSETS = np.array(_r2_offsets_list, dtype=COORD_DTYPE)

@njit(cache=True)
def _compute_auras(
    occupied_types: np.ndarray, 
    occupied_coords: np.ndarray, 
    occupied_colors: np.ndarray,
    count: int,
    active_color: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Buff, Debuff, and Freeze maps for the board.
    Returns: (is_buffed, is_debuffed, is_frozen)
    """
    is_buffed = np.zeros((SIZE, SIZE, SIZE), dtype=np.bool_)
    is_debuffed = np.zeros((SIZE, SIZE, SIZE), dtype=np.bool_)
    is_frozen = np.zeros((SIZE, SIZE, SIZE), dtype=np.bool_)
    
    # Piece Types (Values checked against shared_types.py)
    # SPEEDER=29, SLOWER=30, FREEZER=23, REFLECTOR=35
    SPEEDER = 29
    SLOWER = 30
    FREEZER = 23
    REFLECTOR = 35
    
    # Pre-compute enemy color
    enemy_color = 2 if active_color == 1 else 1
    
    for i in range(count):
        ptype = occupied_types[i]
        
        # Optimization: Check if type is one of the aura providers
        if ptype != SPEEDER and ptype != SLOWER and ptype != FREEZER and ptype != REFLECTOR:
            continue
            
        p_color = occupied_colors[i]
        px, py, pz = occupied_coords[i]
        
        # 1. SPEEDER (Buff Friendly Radius 1)
        if ptype == SPEEDER:
            # Applies range buff to FRIENDLY pieces within radius 1
            if p_color == active_color:
                for k in range(KING_MOVEMENT_VECTORS.shape[0]):
                    dx, dy, dz = KING_MOVEMENT_VECTORS[k]
                    tx, ty, tz = px + dx, py + dy, pz + dz
                    if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                        is_buffed[tx, ty, tz] = True

        # 2. SLOWER (Debuff Enemy Radius 2)
        elif ptype == SLOWER:
            # Applies range debuff to ENEMY pieces
            # If p_color == enemy_of_active (i.e. p_color != active_color), it debuffs US
            if p_color != active_color:
                for k in range(RADIUS_2_OFFSETS.shape[0]):
                    dx, dy, dz = RADIUS_2_OFFSETS[k]
                    tx, ty, tz = px + dx, py + dy, pz + dz
                    if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                        is_debuffed[tx, ty, tz] = True
                        
        # 3. FREEZER (Freeze Enemy Radius 2)
        elif ptype == FREEZER:
            # Freezes ENEMY pieces
            # If p_color != active_color, it freezes US
            if p_color != active_color:
                for k in range(RADIUS_2_OFFSETS.shape[0]):
                    dx, dy, dz = RADIUS_2_OFFSETS[k]
                    tx, ty, tz = px + dx, py + dy, pz + dz
                    if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                        is_frozen[tx, ty, tz] = True

        # 4. REFLECTOR (Buff Friendly Neighbors)
        # Reflector uses Radius 1 (King) logic from previous implementation
        elif ptype == REFLECTOR:
            if p_color == active_color:
                for k in range(KING_MOVEMENT_VECTORS.shape[0]):
                    dx, dy, dz = KING_MOVEMENT_VECTORS[k]
                    tx, ty, tz = px + dx, py + dy, pz + dz
                    if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                        # Reflector also counts as "Buff" in this simplified bool map?
                        # Yes, we merged Reflector buff into is_buffed
                        is_buffed[tx, ty, tz] = True

    return is_buffed, is_debuffed, is_frozen


@njit(cache=True)
def _compute_range_mods(pos: np.ndarray, is_buffed: np.ndarray, is_debuffed: np.ndarray) -> np.ndarray:
    """Batch compute range modifiers (-1/0/+1) for positions."""
    n = pos.shape[0]
    mods = np.empty(n, dtype=np.int32)
    for i in range(n):
        x, y, z = pos[i, 0], pos[i, 1], pos[i, 2]
        b = 1 if is_buffed[x, y, z] else 0
        d = 1 if is_debuffed[x, y, z] else 0
        mods[i] = b - d
    return mods

# --- 4. Main Generator ---

@njit(cache=True)
def _group_indices(types: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group active piece indices by type."""
    # Count per type
    # types are 0..N_PIECE_TYPES
    # We use a safe upper bound e.g. 64 or N_PIECE_TYPES+1
    max_type = 64 
    counts = np.zeros(max_type, dtype=np.int32)
    for i in range(count):
        counts[types[i]] += 1
        
    offsets = np.zeros(max_type, dtype=np.int32)
    current = 0
    for i in range(max_type):
        offsets[i] = current
        current += counts[i]
        
    sorted_indices = np.empty(count, dtype=np.int32)
    current_offsets = offsets.copy()
    
    for i in range(count):
        t = types[i]
        pos = current_offsets[t]
        sorted_indices[pos] = i
        current_offsets[t] += 1
        
    return sorted_indices, offsets, counts

@njit(cache=True)
def generate_moves(buffer: GameBuffer) -> np.ndarray:
    """
    Generate ALL pseudo-legal moves for the active color.
    Returns (N, 6) array.
    """
    return generate_moves_subset(buffer, np.arange(buffer.occupied_count).astype(np.int32))

from typing import Optional, Tuple # Added for type hints
@njit(cache=True, nogil=True)
def generate_moves_subset(
    buffer: GameBuffer, 
    subset_indices: np.ndarray,
    scratch_pad: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    Generate moves for a subset of pieces, using full buffer for context (auras).
    
    Args:
        buffer: Game state buffer
        subset_indices: Indices of pieces to generate moves for
        scratch_pad: Optional tuple of reusable buffers (type_counts, effective_types, sorted_indices, offsets)
                     to reuse memory and avoid allocations.
    """
    active_color = buffer.meta[0]
    
    # Unpack Buffer Arrays
    occ_coords = buffer.occupied_coords
    occ_types = buffer.occupied_types
    occ_colors = buffer.occupied_colors
    occ_count = buffer.occupied_count
    
    board_color = buffer.board_color
    board_type = buffer.board_type
    board_color_flat = buffer.board_color_flat
    
    # 1. Check Buffs (Cached in Buffer)
    is_buffed = buffer.is_buffed
    is_debuffed = buffer.is_debuffed
    is_frozen = buffer.is_frozen
    
    # _compute_auras is no longer needed (O(N) Saved!)
    
    # 2. Filter Active Pieces (Subset)
    if subset_indices.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    subset_count = subset_indices.size
    max_type = 64
    
    # Resolve Scratch Buffers
    if scratch_pad is not None:
        type_counts, effective_types, sorted_indices, offsets = scratch_pad
        # Reset counts and offsets (others are overwritten or slicing handles it)
        type_counts.fill(0)
        # Note: effective_types and sorted_indices are large enough buffers, we slice them
        eff_types_view = effective_types[:subset_count]
        sorted_idx_view = sorted_indices[:subset_count]
    else:
        type_counts = np.zeros(max_type, dtype=np.int32)
        effective_types = np.empty(subset_count, dtype=np.int32)
        sorted_indices = np.empty(subset_count, dtype=np.int32)
        offsets = np.zeros(max_type, dtype=np.int32)
        eff_types_view = effective_types
        sorted_idx_view = sorted_indices
    
    for k in range(subset_count):
        idx = subset_indices[k]
        
        # Check Active Color (Safety)
        if occ_colors[idx] != active_color:
            eff_types_view[k] = -1 # Skip
            continue
            
        px, py, pz = occ_coords[idx]
        
        # Check Frozen
        if is_frozen[px, py, pz]:
            eff_types_view[k] = -1 # Skip
            continue
            
        t = occ_types[idx]
        # Check Debuff (Turns into PAWN=1)
        if is_debuffed[px, py, pz]:
            t = 1
            
        eff_types_view[k] = t
        if t < max_type:
            type_counts[t] += 1
            
    # Calculate Offsets
    # If using scratch pad, offsets array is passed in, we just compute into it
    # We must ensure we zero it out first if we accum (but logical accum doesn't need zero if we overwrite)
    # The loop overwrites offsets[i] completely.
    cur = 0
    for i in range(max_type):
        offsets[i] = cur
        cur += type_counts[i]
        
    # Build Sorted Indices
    current_offsets = offsets.copy()
    
    for k in range(subset_count):
        t = eff_types_view[k]
        if t != -1 and t < max_type:
            pos = current_offsets[t]
            sorted_idx_view[pos] = subset_indices[k] # Store GLOBAL index
            current_offsets[t] += 1
            
    # Use the VIEW for downstream logic? 
    # Logic uses `sorted_indices` and `offsets` and `type_counts`.
    # `sorted_indices` variable name now points to potentially full buffer.
    # We should use `sorted_idx_view` for reading if we rely on slicing?
    # BUT downstream code accesses `sorted_indices[start : start+count]`.
    # `start` comes from `offsets`. `bytes` indices are absolute.
    # `sorted_indices` buffer content is valid at those indices.
    # So using `sorted_indices` (buffer) is fine!
    
    # 3. Iterate Types and Generate
    # 3. Iterate Types and Generate
    # Initialize with typed dummy to help Numba inference
    dummy_moves = np.empty((0, 6), dtype=COORD_DTYPE)
    results_list = [dummy_moves]
    # Numba doesn't support list of arrays well for concatenation?
    # It supports np.concatenate((a, b)) but list support is limited.
    # We can pre-calculate max moves or use a dynamic builder?
    # Better: Use a fixed large buffer or multiple passes?
    # Dynamic list of arrays is supported in recent Numba if typed.
    # Let's try simple list append.
    
    # PAWN (1)
    pawn_count = type_counts[1]
    if pawn_count > 0:
        start = offsets[1]
        indices = sorted_indices[start : start+pawn_count]
        pawn_pos = occ_coords[indices] # Fancy indexing copy
        
        # Params
        dz = 1 if active_color == 1 else -1
        start_rank = PAWN_START_RANK_WHITE if active_color == 1 else PAWN_START_RANK_BLACK
        atk_dirs = PAWN_ATTACK_DIRECTIONS_WHITE if active_color == 1 else PAWN_ATTACK_DIRECTIONS_BLACK
        
        moves = _generate_pawn_moves_batch_kernel(
            pawn_pos, board_color, board_type, active_color,
            start_rank, dz, atk_dirs, 9 # Armour type assumed 9 or use shared_types
        )
        if moves.size > 0:
            results_list.append(moves)

    # KNIGHT (2)
    knight_count = type_counts[2]
    if knight_count > 0:
        start = offsets[2]
        indices = sorted_indices[start : start+knight_count]
        pos = occ_coords[indices]
        
        moves = _generate_and_filter_jump_moves_batch(
            pos, KNIGHT_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if moves.size > 0:
            results_list.append(moves)

    # BISHOP (3)
    bishop_count = type_counts[3]
    if bishop_count > 0:
        start = offsets[3]
        indices = sorted_indices[start : start+bishop_count]
        pos = occ_coords[indices]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = SIZE_MINUS_1 + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, BISHOP_MOVEMENT_VECTORS, max_dists, board_color, False
        )
        if moves.size > 0:
            results_list.append(moves)

    # ROOK (4)
    rook_count = type_counts[4]
    if rook_count > 0:
        start = offsets[4]
        indices = sorted_indices[start : start+rook_count]
        pos = occ_coords[indices]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = SIZE_MINUS_1 + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, ROOK_MOVEMENT_VECTORS, max_dists, board_color, False
        )
        if moves.size > 0:
            results_list.append(moves)

    # QUEEN (5)
    queen_count = type_counts[5]
    if queen_count > 0:
        start = offsets[5]
        indices = sorted_indices[start : start+queen_count]
        pos = occ_coords[indices]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = SIZE_MINUS_1 + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, QUEEN_MOVEMENT_VECTORS, max_dists, board_color, False
        )
        if moves.size > 0:
            results_list.append(moves)

    # KING (6)
    king_count = type_counts[6]
    if king_count > 0:
        start = offsets[6]
        indices = sorted_indices[start : start+king_count]
        pos = occ_coords[indices]
        
        # King uses jump engine
        moves = _generate_and_filter_jump_moves_batch(
            pos, KING_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if moves.size > 0:
            results_list.append(moves)

    # PRIEST (7) - Uses King Vectors
    priest_count = type_counts[7]
    if priest_count > 0:
        start = offsets[7]
        indices = sorted_indices[start : start+priest_count]
        pos = occ_coords[indices]
        
        moves = _generate_and_filter_jump_moves_batch(
            pos, KING_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if moves.size > 0:
            results_list.append(moves)
            
    # WALL (34 ? Check shared_types)
    # PieceType.WALL is 24 usually (checked shared_types.py).
    # Correcting Types based on shared_types.py
    # PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
    # PRIEST=7, KNIGHT32=8, KNIGHT31=9, TRIGONALBISHOP=10
    # ORBITER=12
    # PANEL=15
    # EDGEROOK=16, XYQUEEN=17, XZQUEEN=18, YZQUEEN=19
    # WALL=24
    
    # 8. KNIGHT32 (Zebra)
    if type_counts[8] > 0:
        start = offsets[8]
        indices = sorted_indices[start : start+type_counts[8]]
        pos = occ_coords[indices]
        moves = _generate_and_filter_jump_moves_batch(
            pos, KNIGHT32_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if moves.size > 0: results_list.append(moves)

    # 9. KNIGHT31 (Camel)
    if type_counts[9] > 0:
        start = offsets[9]
        indices = sorted_indices[start : start+type_counts[9]]
        pos = occ_coords[indices]
        moves = _generate_and_filter_jump_moves_batch(
            pos, KNIGHT31_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if moves.size > 0: results_list.append(moves)

    # 10. TRIGONAL BISHOP
    if type_counts[10] > 0:
        start = offsets[10]
        indices = sorted_indices[start : start+type_counts[10]]
        pos = occ_coords[indices]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = SIZE_MINUS_1 + mods # (Clamping handled by engine or max valid coord)
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, TRIGONAL_BISHOP_VECTORS, max_dists, board_color, False
        )
        if moves.size > 0: results_list.append(moves)

    # 12. ORBITER
    if type_counts[12] > 0:
        start = offsets[12]
        indices = sorted_indices[start : start+type_counts[12]]
        pos = occ_coords[indices]
        # Check buffs locally or via pre-calcbuffs?
        # Assuming unbuffed for now unless we integrate consolidated aura
        # TODO: Integrate buffed directions (Orbiter uses different radius)
        # For now, using unbuffed or buffed based on _compute_buffs mask check?
        # The logic below handles mix if we iterate, but batch handles one set.
        # Let's support both if we split. For now, try Standard (Unbuffed)
        moves = _generate_and_filter_jump_moves_batch(
            pos, _ORBITAL_DIRS, board_color, True, active_color
        )
        if moves.size > 0: results_list.append(moves)

    # 16. EDGEROOK
    if type_counts[16] > 0:
        start = offsets[16]
        indices = sorted_indices[start : start+type_counts[16]]
        pos = occ_coords[indices]
        
        # Need flattened occupancy for this kernel (legacy artifact)
        # We can construct it or kernel accepts coords? 
        # _edgerook_bfs_batch_numba accepts flattened_occ array (SIZE^3).
        # We have board_color (dense grid). We can flatten it.
        start_nodes = pos.astype(COORD_DTYPE)
        flat_occ = board_color_flat
        
        # Numba kernel call
        moves = _edgerook_bfs_batch_numba(start_nodes, _EDGE_NEIGHBORS, flat_occ, active_color)
        if moves.size > 0: results_list.append(moves)

    # 17. XYQUEEN
    if type_counts[17] > 0:
        start = offsets[17]
        indices = sorted_indices[start : start+type_counts[17]]
        pos = occ_coords[indices]
        cnt = type_counts[17]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = SIZE_MINUS_1 + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        m1, _ = _generate_all_slider_moves_batch(
            active_color, pos, _XY_SLIDER_DIRS, max_dists, board_color, False
        )
        m2 = _generate_and_filter_jump_moves_batch(
            pos, _KING_3D_DIRS, board_color, True, active_color
        )
        if m1.size > 0: results_list.append(m1)
        if m2.size > 0: results_list.append(m2)

    # 18. XZQUEEN
    if type_counts[18] > 0:
        start = offsets[18]
        indices = sorted_indices[start : start+type_counts[18]]
        pos = occ_coords[indices]
        cnt = type_counts[18]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = SIZE_MINUS_1 + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        m1, _ = _generate_all_slider_moves_batch(
            active_color, pos, _XZ_SLIDER_DIRS, max_dists, board_color, False
        )
        m2 = _generate_and_filter_jump_moves_batch(
            pos, _KING_3D_DIRS, board_color, True, active_color
        )
        if m1.size > 0: results_list.append(m1)
        if m2.size > 0: results_list.append(m2)
    
    # 19. YZQUEEN
    if type_counts[19] > 0:
        start = offsets[19]
        indices = sorted_indices[start : start+type_counts[19]]
        pos = occ_coords[indices]
        cnt = type_counts[19]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = SIZE_MINUS_1 + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        m1, _ = _generate_all_slider_moves_batch(
            active_color, pos, _YZ_SLIDER_DIRS, max_dists, board_color, False
        )
        m2 = _generate_and_filter_jump_moves_batch(
            pos, _KING_3D_DIRS, board_color, True, active_color
        )
        if m1.size > 0: results_list.append(m1)
        if m2.size > 0: results_list.append(m2)

    # 20. VECTORSLIDER (Check shared_types, might be different ID)
    # shared_types says VECTORSLIDER=20
    # 20. VECTORSLIDER (Check shared_types, might be different ID)
    # shared_types says VECTORSLIDER=20
    if type_counts[20] > 0:
        start = offsets[20]
        indices = sorted_indices[start : start+type_counts[20]]
        pos = occ_coords[indices]
        cnt = type_counts[20]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = 8 + mods  # max distance 8
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, VECTOR_DIRECTIONS, max_dists, board_color, False
        )
        if moves.size > 0: results_list.append(moves)

    # 21. CONESLIDER (FaceCone)
    if type_counts[21] > 0:
        start = offsets[21]
        indices = sorted_indices[start : start+type_counts[21]]
        pos = occ_coords[indices]
        cnt = type_counts[21]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = SIZE_MINUS_1 + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, FACE_CONE_MOVEMENT_VECTORS, max_dists, board_color, False
        )
        if moves.size > 0: results_list.append(moves)

    # 22. MIRROR
    if type_counts[22] > 0:
        start = offsets[22]
        indices = sorted_indices[start : start+type_counts[22]]
        pos = occ_coords[indices]
        
        # 1. King Moves
        m1 = _generate_and_filter_jump_moves_batch(
            pos, KING_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if m1.size > 0: results_list.append(m1)
        
        # 2. Mirror Teleport
        flat_occ = board_color_flat
        
        # Use simple mods loop to check if buffed
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        # mod >= 1 means buffed
        
        # We need to split into batches for kernel
        # Can we avoid list loop?
        # Numba supports boolean masks
        
        # Auras logic: speeder buffs range (so mod +1).
        # Mirror logic: mirror all axes if "Buffed". 
        # Usually checking 'is_buffed' map directly is safer than range mod.
        # But we don't have is_buffed map here easily (it's in closure context if passed?)
        # We passed it to _compute_range_mods.
        # Let's duplicate the lookup logic here since we need mask.
        
        buffed_mask = np.empty(type_counts[22], dtype=np.bool_)
        for k in range(type_counts[22]):
            px, py, pz = pos[k,0], pos[k,1], pos[k,2]
            buffed_mask[k] = is_buffed[px, py, pz]
            
        unbuffed_mask = ~buffed_mask
        
        if np.any(unbuffed_mask):
            ub_pos = pos[unbuffed_mask]
            m_ub = _generate_mirror_moves_kernel(
                ub_pos.astype(COORD_DTYPE), flat_occ, active_color, False
            )
            if m_ub.size > 0: results_list.append(m_ub)
            
        if np.any(buffed_mask):
            b_pos = pos[buffed_mask]
            m_b = _generate_mirror_moves_kernel(
                b_pos.astype(COORD_DTYPE), flat_occ, active_color, True
            )
            if m_b.size > 0: results_list.append(m_b)

    # --- SLIDERS ---

    # 33. XZ ZIGZAG
    if type_counts[33] > 0:
        start = offsets[33]
        indices = sorted_indices[start : start+type_counts[33]]
        pos = occ_coords[indices]
        cnt = type_counts[33]
        max_dists = np.full(cnt, 16, dtype=np.int32)
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, XZ_ZIGZAG_DIRECTIONS, max_dists, board_color, False
        )
        if moves.size > 0: results_list.append(moves)

    # 34. YZ ZIGZAG
    if type_counts[34] > 0:
        start = offsets[34]
        indices = sorted_indices[start : start+type_counts[34]]
        pos = occ_coords[indices]
        cnt = type_counts[34]
        max_dists = np.full(cnt, 16, dtype=np.int32)
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, YZ_ZIGZAG_DIRECTIONS, max_dists, board_color, False
        )
        if moves.size > 0: results_list.append(moves)

    # 35. REFLECTOR
    if type_counts[35] > 0:
        start = offsets[35]
        indices = sorted_indices[start : start+type_counts[35]]
        pos = occ_coords[indices] # coords
        
        # Need flattened board?? No, check key signature in reflector.py
        # _trace_reflector_rays_batch(occupancy: flat, origins, dirs, max_bounces, color, ignore)
        # It needs FLATTENED occupancy.
        flat_occ = board_color_flat
        
        moves = _trace_reflector_rays_batch(
            flat_occ, pos.astype(COORD_DTYPE), _REFLECTOR_DIRS, 2, active_color, False
        )
        if moves.size > 0: results_list.append(moves)

    # 39. TRAILBLAZER
    if type_counts[39] > 0:
        start = offsets[39]
        indices = sorted_indices[start : start+type_counts[39]]
        pos = occ_coords[indices]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = MAX_TRAILBLAZER_DISTANCE + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, ROOK_DIRECTIONS, max_dists, board_color, False
        )
        if moves.size > 0: results_list.append(moves)

    # 40. SPIRAL
    if type_counts[40] > 0:
        start = offsets[40]
        indices = sorted_indices[start : start+type_counts[40]]
        pos = occ_coords[indices]
        
        mods = _compute_range_mods(pos, is_buffed, is_debuffed)
        max_dists = MAX_SPIRAL_DISTANCE + mods
        max_dists = np.maximum(max_dists, 1).astype(np.int32)
        
        moves, _ = _generate_all_slider_moves_batch(
            active_color, pos, SPIRAL_MOVEMENT_VECTORS, max_dists, board_color, False
        )
        if moves.size > 0: results_list.append(moves)

    # --- SPECIAL COMPLEX ---

    # 38. INFILTRATOR
    if type_counts[38] > 0:
        start = offsets[38]
        indices = sorted_indices[start : start+type_counts[38]]
        pos = occ_coords[indices]
        
        # 1. King Moves
        m1 = _generate_and_filter_jump_moves_batch(
            pos, KING_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if m1.size > 0: results_list.append(m1)
        
        # 2. Teleport
        # Split buffed/unbuffed for direction
        cnt_inf = type_counts[38]
        is_buffed_mask = np.empty(cnt_inf, dtype=np.bool_)
        for k in range(cnt_inf):
            idx = indices[k]
            px, py, pz = occ_coords[idx]
            is_buffed_mask[k] = is_buffed[px, py, pz]
        
        unbuffed_mask = ~is_buffed_mask
        opp_color = 2 if active_color == 1 else 1

        if np.any(unbuffed_mask):
            ub_pos = pos[unbuffed_mask]
            # Front
            dz_front = -1 if opp_color == 2 else 1
            targets = _get_pawn_targets_kernel(board_color, board_type, opp_color, dz_front)
            if targets.shape[0] > 0:
                m_ub = _generate_infiltrator_teleport_moves(ub_pos.astype(COORD_DTYPE), targets)
                if m_ub.size > 0: results_list.append(m_ub)
        
        if np.any(is_buffed_mask):
            b_pos = pos[is_buffed_mask]
            # Behind
            dz_behind = 1 if opp_color == 2 else -1
            targets = _get_pawn_targets_kernel(board_color, board_type, opp_color, dz_behind)
            if targets.shape[0] > 0:
                m_b = _generate_infiltrator_teleport_moves(b_pos.astype(COORD_DTYPE), targets)
                if m_b.size > 0: results_list.append(m_b)

    # --- GROUPED KING-LIKE UTILITY ---
    # Simple king-like pieces (no special ability beyond king moves):
    # 11=HIVE, 13=NEBULA, 15=PANEL, 23=FREEZER, 25=ARCHER, 26=BOMB, 
    # 28=ARMOUR, 29=SPEEDER, 30=SLOWER, 36=BLACKHOLE, 37=WHITEHOLE
    simple_king_types = (11, 13, 15, 23, 25, 26, 28, 29, 30, 36, 37)
    
    for kt in simple_king_types:
        if kt < max_type and type_counts[kt] > 0:
            start = offsets[kt]
            indices = sorted_indices[start : start+type_counts[kt]]
            pos = occ_coords[indices]
            moves = _generate_and_filter_jump_moves_batch(
                pos, KING_MOVEMENT_VECTORS, board_color, True, active_color
            )
            if moves.size > 0: results_list.append(moves)
            
    # 14. ECHO (not simple king - uses anchor+bubble directions)
    if type_counts[14] > 0:
        start = offsets[14]
        indices = sorted_indices[start : start+type_counts[14]]
        pos = occ_coords[indices]
        # Use echo directions (156 vectors)
        moves = _generate_and_filter_jump_moves_batch(
            pos, _ECHO_DIRECTIONS, board_color, True, active_color
        )
        if moves.size > 0: results_list.append(moves)

    # 27. FRIENDLYTELEPORTER - Teleport to king-move destinations from ANY friendly piece
    # Rule: Can teleport to any square that a friendly piece could move to with a king move
    #       (cannot land on friendly, can capture enemies)
    if type_counts[27] > 0:
        start = offsets[27]
        indices = sorted_indices[start : start+type_counts[27]]
        tp_positions = occ_coords[indices]
        
        # 1. Find all friendly pieces
        friendly_coords = np.empty((occ_count, 3), dtype=COORD_DTYPE)
        fc_idx = 0
        for i in range(occ_count):
            if occ_colors[i] == active_color:
                friendly_coords[fc_idx] = occ_coords[i]
                fc_idx += 1
        friendly_coords = friendly_coords[:fc_idx]
        
        # 2. Build set of valid teleport destinations (king moves from each friendly)
        n_dirs = KING_MOVEMENT_VECTORS.shape[0]
        # Max destinations = fc_idx * 26, deduplicated later
        dest_mask = np.zeros((SIZE, SIZE, SIZE), dtype=np.bool_)
        
        for i in range(fc_idx):
            fx, fy, fz = friendly_coords[i]
            for d in range(n_dirs):
                dx, dy, dz = KING_MOVEMENT_VECTORS[d]
                tx, ty, tz = fx + dx, fy + dy, fz + dz
                if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                    # Can't land on friendly piece
                    if board_color[tx, ty, tz] != active_color:
                        dest_mask[tx, ty, tz] = True
        
        # 3. Generate moves from each TP to each valid destination
        n_tps = tp_positions.shape[0]
        dest_count = np.sum(dest_mask)
        max_moves = n_tps * dest_count
        moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
        count = 0
        
        for x in range(SIZE):
            for y in range(SIZE):
                for z in range(SIZE):
                    if dest_mask[x, y, z]:
                        for i in range(n_tps):
                            sx, sy, sz = tp_positions[i]
                            # Skip if TP is already at destination
                            if sx == x and sy == y and sz == z:
                                continue
                            moves[count, 0] = sx
                            moves[count, 1] = sy
                            moves[count, 2] = sz
                            moves[count, 3] = x
                            moves[count, 4] = y
                            moves[count, 5] = z
                            count += 1
        
        if count > 0:
            results_list.append(moves[:count])

    # 31. GEOMANCER - King moves + Geomancy placement (radius 2-3 or 2-4)
    if type_counts[31] > 0:
        start = offsets[31]
        indices = sorted_indices[start : start+type_counts[31]]
        pos = occ_coords[indices]
        
        # King moves
        m1 = _generate_and_filter_jump_moves_batch(
            pos, KING_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if m1.size > 0: results_list.append(m1)
        
        # Geomancy placement (need to iterate due to buff check)
        flat_occ = board_color_flat
        for k in range(type_counts[31]):
            p = pos[k:k+1]
            px, py, pz = pos[k]
            gm_buffed = is_buffed[px, py, pz]
            offsets_to_use = BUFFED_GEOMANCY_OFFSETS if gm_buffed else GEOMANCY_OFFSETS
            gm_moves = _generate_geomancy_moves_kernel(p, flat_occ, offsets_to_use)
            if gm_moves.size > 0: results_list.append(gm_moves)
            
    # 32. SWAPPER - King moves + Swap to ANY friendly piece
    if type_counts[32] > 0:
        start = offsets[32]
        indices = sorted_indices[start : start+type_counts[32]]
        pos = occ_coords[indices]
        
        # King moves
        m1 = _generate_and_filter_jump_moves_batch(
            pos, KING_MOVEMENT_VECTORS, board_color, True, active_color
        )
        if m1.size > 0: results_list.append(m1)
        
        # Swap moves - get all friendly positions and types
        friendly_mask = (occ_colors[:occ_count] == active_color)
        friendly_count = np.sum(friendly_mask)
        friendly_coords = np.empty((friendly_count, 3), dtype=COORD_DTYPE)
        friendly_types = np.empty(friendly_count, dtype=np.int8)
        fc_idx = 0
        for i in range(occ_count):
            if occ_colors[i] == active_color:
                friendly_coords[fc_idx] = occ_coords[i]
                friendly_types[fc_idx] = occ_types[i]
                fc_idx += 1
        
        swap_moves = _generate_swap_moves_kernel(pos, friendly_coords, friendly_types)
        if swap_moves.size > 0: results_list.append(swap_moves)

    # 24. WALL (from shared_types)
    WALL_TYPE = 24
    if WALL_TYPE < max_type:
        wall_count = type_counts[WALL_TYPE]
        if wall_count > 0:
            start = offsets[WALL_TYPE]
            indices = sorted_indices[start : start+wall_count]
            pos = occ_coords[indices]
            
            UNBUFFED_WALL_VECTORS = np.array([
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]
            ], dtype=COORD_DTYPE)
            
            valid_indices = _filter_anchors_numba(pos, board_type)
            if valid_indices.shape[0] > 0:
                 valid_full_indices = indices[valid_indices]
                 valid_pos = occ_coords[valid_full_indices]
                 
                 buff_status = np.empty(valid_pos.shape[0], dtype=np.int8) 
                 for k in range(valid_pos.shape[0]):
                     wx, wy, wz = valid_pos[k]
                     buff_status[k] = 1 if is_buffed[wx, wy, wz] else 0
                 
                 moves = _generate_wall_moves_fused_kernel(
                    valid_pos, board_color, active_color, SIZE,
                    UNBUFFED_WALL_VECTORS, BUFFED_KING_MOVEMENT_VECTORS,
                    buff_status 
                 )
                 if moves.size > 0:
                    results_list.append(moves)


    # Concatenate
    if len(results_list) == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    # Manual concat if needed, or simplevstack
    # Just calculate total size
    total_len = 0
    for res in results_list:
        total_len += res.shape[0]
        
    final_moves = np.empty((total_len, 6), dtype=COORD_DTYPE)
    curr = 0
    for res in results_list:
        cnt = res.shape[0]
        final_moves[curr : curr+cnt] = res
        curr += cnt
        
    # Defensive bounds validation
    if total_len > 0:
        valid_mask = (
            (final_moves[:, 3] >= 0) & (final_moves[:, 3] < SIZE) &
            (final_moves[:, 4] >= 0) & (final_moves[:, 4] < SIZE) &
            (final_moves[:, 5] >= 0) & (final_moves[:, 5] < SIZE)
        )
        if not np.all(valid_mask):
            # Filter OOB
            final_moves = final_moves[valid_mask]
        
    return final_moves
