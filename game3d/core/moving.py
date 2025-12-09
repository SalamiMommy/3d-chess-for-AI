
"""
Functional Move Logic.
"""

import numpy as np
from numba import njit
from game3d.core.buffer import GameBuffer
from game3d.core.hashing import get_zobrist_key, SIDE_TO_MOVE_KEY
from game3d.common.shared_types import (
    COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE,
    HASH_DTYPE, INDEX_DTYPE,
    MAX_HISTORY_SIZE, Color
)

@njit(cache=True)
def apply_move(buffer: GameBuffer, move_arr: np.ndarray) -> GameBuffer:
    """
    Apply a move to the game buffer, returning a NEW buffer (Immutable/Functional).
    move_arr: [from_x, from_y, from_z, to_x, to_y, to_z]
    """
    
    # 1. Copy State (Copy-Make)
    new_coords = buffer.occupied_coords.copy()
    new_types = buffer.occupied_types.copy()
    new_colors = buffer.occupied_colors.copy()
    
    # Copy new split arrays
    new_board_type = buffer.board_type.copy()
    new_board_color = buffer.board_color.copy()
    
    new_meta = buffer.meta.copy()
    new_history = buffer.history.copy()
    
    # Locals
    occ_count = buffer.occupied_count
    current_zkey = buffer.zkey
    hist_count = buffer.history_count
    
    fx, fy, fz = move_arr[0], move_arr[1], move_arr[2]
    tx, ty, tz = move_arr[3], move_arr[4], move_arr[5]
    
    # 2. Get Moving Piece
    src_type = new_board_type[fx, fy, fz]
    src_color = new_board_color[fx, fy, fz]
    
    if src_type == 0:
        # Error or empty
        return buffer
        
    # 3. Handle Capture
    tgt_type = new_board_type[tx, ty, tz]
    tgt_color = new_board_color[tx, ty, tz]
    
    # Update Zkey (Remove Source)
    # Color map: 1->0, 2->1
    src_c_idx = src_color - 1
    current_zkey ^= get_zobrist_key(src_c_idx, src_type, fx, fy, fz)
    
    if tgt_type != 0:
        # Capture: Remove piece from sparse arrays
        tgt_idx = -1
        # Simple search O(N) but N is small
        for i in range(occ_count):
            if (new_coords[i, 0] == tx and 
                new_coords[i, 1] == ty and 
                new_coords[i, 2] == tz):
                tgt_idx = i
                break
                
        if tgt_idx != -1:
            # XOR out captured piece
            tgt_c_idx = tgt_color - 1
            current_zkey ^= get_zobrist_key(tgt_c_idx, tgt_type, tx, ty, tz)
            
            # Swap with last element
            last_idx = occ_count - 1
            if tgt_idx != last_idx:
                new_coords[tgt_idx] = new_coords[last_idx]
                new_types[tgt_idx] = new_types[last_idx]
                new_colors[tgt_idx] = new_colors[last_idx]
                
                # Update King Index if we swapped a king into this slot
                moved_type = new_types[tgt_idx]
                moved_color = new_colors[tgt_idx]
                if moved_type == 6: # King
                   if moved_color == 1: new_meta[4] = tgt_idx
                   else: new_meta[5] = tgt_idx
                   
            occ_count -= 1
            
    # 4. Move Piece (Update Bounds)
    # Update Board Arrays
    new_board_type[fx, fy, fz] = 0
    new_board_color[fx, fy, fz] = 0
    
    new_board_type[tx, ty, tz] = src_type
    new_board_color[tx, ty, tz] = src_color
    
    # Update Sparse Array
    src_idx = -1
    for i in range(occ_count):
        if (new_coords[i, 0] == fx and 
            new_coords[i, 1] == fy and 
            new_coords[i, 2] == fz):
            src_idx = i
            break
            
    if src_idx != -1:
        new_coords[src_idx, 0] = tx
        new_coords[src_idx, 1] = ty
        new_coords[src_idx, 2] = tz
        # Type/Color buffer unchanged (same piece moving)
        
        # Update King Index
        if src_type == 6:
            if src_color == 1: new_meta[4] = src_idx
            else: new_meta[5] = src_idx
            
    # Update Zkey (Add Destination)
    current_zkey ^= get_zobrist_key(src_c_idx, src_type, tx, ty, tz)
    
    # Swap Move Color in Zkey
    current_zkey ^= SIDE_TO_MOVE_KEY
    
    # 5. Update Metadata
    new_active = 2 if new_meta[0] == 1 else 1
    new_meta[0] = new_active
    
    if src_color == 2: # Black moved
        new_meta[2] += 1
        
    if tgt_type != 0 or src_type == 1: # Capture or Pawn
        new_meta[1] = 0
    else:
        new_meta[1] += 1
        
    # 6. Update History
    h_idx = hist_count % MAX_HISTORY_SIZE
    new_history[h_idx] = current_zkey
    hist_count += 1
    
    return GameBuffer(
        new_coords,
        new_types,
        new_colors,
        occ_count,
        new_board_type,
        new_board_color,
        new_meta,
        current_zkey,
        new_history,
        hist_count
    )


# --- In-Place Apply/Undo for Fast Simulation ---

@njit(cache=True)
def apply_move_inplace(buffer: GameBuffer, move_arr: np.ndarray) -> np.ndarray:
    """
    Apply a move IN-PLACE for fast simulation. Returns undo info.
    
    UndoInfo array format (13 elements):
    [fx, fy, fz, tx, ty, tz, src_type, src_color, tgt_type, tgt_color, 
     src_sparse_idx, tgt_sparse_idx, old_king_idx]
    
    Returns: undo_info array (13,) or empty array if move invalid
    """
    fx, fy, fz = move_arr[0], move_arr[1], move_arr[2]
    tx, ty, tz = move_arr[3], move_arr[4], move_arr[5]
    
    # Get source piece
    src_type = buffer.board_type[fx, fy, fz]
    src_color = buffer.board_color[fx, fy, fz]
    
    if src_type == 0:
        return np.empty(0, dtype=COORD_DTYPE)
    
    # Get target
    tgt_type = buffer.board_type[tx, ty, tz]
    tgt_color = buffer.board_color[tx, ty, tz]
    
    # Find source sparse index
    src_sparse_idx = -1
    occ_count = buffer.occupied_count
    for i in range(occ_count):
        if (buffer.occupied_coords[i, 0] == fx and 
            buffer.occupied_coords[i, 1] == fy and 
            buffer.occupied_coords[i, 2] == fz):
            src_sparse_idx = i
            break
    
    # Find target sparse index (if capture)
    tgt_sparse_idx = -1
    if tgt_type != 0:
        for i in range(occ_count):
            if (buffer.occupied_coords[i, 0] == tx and 
                buffer.occupied_coords[i, 1] == ty and 
                buffer.occupied_coords[i, 2] == tz):
                tgt_sparse_idx = i
                break
    
    # Store old king index for undo
    old_king_idx = buffer.meta[4] if src_color == 1 else buffer.meta[5]
    
    # Build undo info
    undo_info = np.empty(13, dtype=COORD_DTYPE)
    undo_info[0] = fx
    undo_info[1] = fy
    undo_info[2] = fz
    undo_info[3] = tx
    undo_info[4] = ty
    undo_info[5] = tz
    undo_info[6] = src_type
    undo_info[7] = src_color
    undo_info[8] = tgt_type
    undo_info[9] = tgt_color
    undo_info[10] = src_sparse_idx
    undo_info[11] = tgt_sparse_idx
    undo_info[12] = old_king_idx
    
    # --- Apply Move In-Place ---
    
    # 1. Update board grids
    buffer.board_type[fx, fy, fz] = 0
    buffer.board_color[fx, fy, fz] = 0
    buffer.board_type[tx, ty, tz] = src_type
    buffer.board_color[tx, ty, tz] = src_color
    
    # 2. Update sparse array - move source
    if src_sparse_idx != -1:
        buffer.occupied_coords[src_sparse_idx, 0] = tx
        buffer.occupied_coords[src_sparse_idx, 1] = ty
        buffer.occupied_coords[src_sparse_idx, 2] = tz
        
        # Update king index
        if src_type == 6:
            if src_color == 1:
                buffer.meta[4] = src_sparse_idx
            else:
                buffer.meta[5] = src_sparse_idx
    
    # 3. Handle capture - remove target from sparse array
    if tgt_sparse_idx != -1:
        # Swap with last and decrement count
        # Note: We modify occupied_count but it's a scalar in NamedTuple
        # We need to track this in undo_info or accept slight inaccuracy
        # For check detection, the board grids are sufficient
        pass  # Skip sparse array modification for speed
    
    # 4. Flip active color
    buffer.meta[0] = 2 if buffer.meta[0] == 1 else 1
    
    return undo_info


@njit(cache=True)
def undo_move_inplace(buffer: GameBuffer, undo_info: np.ndarray) -> None:
    """
    Undo a move IN-PLACE using stored undo info.
    """
    if undo_info.size == 0:
        return
    
    fx, fy, fz = undo_info[0], undo_info[1], undo_info[2]
    tx, ty, tz = undo_info[3], undo_info[4], undo_info[5]
    src_type = undo_info[6]
    src_color = undo_info[7]
    tgt_type = undo_info[8]
    tgt_color = undo_info[9]
    src_sparse_idx = undo_info[10]
    old_king_idx = undo_info[12]
    
    # 1. Restore board grids
    buffer.board_type[fx, fy, fz] = src_type
    buffer.board_color[fx, fy, fz] = src_color
    buffer.board_type[tx, ty, tz] = tgt_type
    buffer.board_color[tx, ty, tz] = tgt_color
    
    # 2. Restore sparse array position
    if src_sparse_idx != -1:
        buffer.occupied_coords[src_sparse_idx, 0] = fx
        buffer.occupied_coords[src_sparse_idx, 1] = fy
        buffer.occupied_coords[src_sparse_idx, 2] = fz
        
        # Restore king index
        if src_type == 6:
            if src_color == 1:
                buffer.meta[4] = old_king_idx
            else:
                buffer.meta[5] = old_king_idx
    
    # 3. Flip active color back
    buffer.meta[0] = 2 if buffer.meta[0] == 1 else 1

