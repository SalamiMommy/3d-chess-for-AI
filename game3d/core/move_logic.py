
"""
Pure functional move logic.
Calculates the effects of a move (coordinates to clear, update, metadata changes)
without modifying the state.
"""

import numpy as np
from typing import Tuple, NamedTuple, Optional
from game3d.common.shared_types import (
    PieceType, COORD_DTYPE, PIECE_TYPE_DTYPE, 
    SIZE, RADIUS_2_OFFSETS
)
from game3d.core.buffer import GameBuffer
from game3d.core.structures import StructureManager
from game3d.common.coord_utils import in_bounds_vectorized

class MoveEffects(NamedTuple):
    coords_to_clear: np.ndarray # (N, 3)
    coords_to_update: np.ndarray # (M, 3)
    new_pieces_data: np.ndarray # (M, 2) [type, color]
    metadata_updates: dict # e.g. {'en_passant': val, 'halfmove': val}

def calculate_move_effects(move: np.ndarray, buffer: GameBuffer) -> MoveEffects:
    """
    Calculate the effects of a move on the board state.
    Args:
        move: (6,) array [fx, fy, fz, tx, ty, tz]
        buffer: Immutable GameBuffer
    """
    from_coord = move[:3]
    to_coord = move[3:]
    
    # Get piece info from buffer using dense array (O(1))
    fx, fy, fz = from_coord
    tx, ty, tz = to_coord
    
    from_type = buffer.board_type[fx, fy, fz]
    from_color = buffer.board_color[fx, fy, fz]
    
    target_type = buffer.board_type[tx, ty, tz]
    target_color = buffer.board_color[tx, ty, tz]
    
    move_distance = np.max(np.abs(to_coord - from_coord))
    
    # Identify Special Moves
    is_archery = (from_type == PieceType.ARCHER and move_distance == 2)
    is_detonation = (from_type == PieceType.BOMB and move_distance == 0)
    
    # Identify Swap (Swapper moving to friendly)
    is_swap = (from_type == PieceType.SWAPPER and 
              target_type != 0 and 
              target_color == from_color)

    # Initialize effects
    coords_to_clear = np.empty((0, 3), dtype=COORD_DTYPE)
    coords_to_update = np.empty((0, 3), dtype=COORD_DTYPE)
    new_pieces_data = np.empty((0, 2), dtype=PIECE_TYPE_DTYPE)
    metadata = {}
    
    if is_archery:
        # Archery: Clear target square (source piece doesn't move)
        # Note: Archery destroys piece at target
        # Atomic structure handling needed? 
        # If target is wall, we must clear whole wall.
        
        target_squares = _get_affected_structure_squares(to_coord, target_type, buffer)
        coords_to_clear = target_squares
        
    elif is_detonation:
        # Bomb: Clear explosion area (Radius 2)
        explosion_offsets = from_coord + RADIUS_2_OFFSETS
        valid_mask = in_bounds_vectorized(explosion_offsets)
        raw_targets = explosion_offsets[valid_mask]
        
        # Include bomb itself (from_coord)
        raw_targets = np.vstack([raw_targets, from_coord.reshape(1, 3)])
        raw_targets = np.unique(raw_targets, axis=0)
        
        # Filter (Don't destroy Kings)
        # We need to check each target for atomic structure and King safety
        final_targets_list = []
        
        # Check each target square
        for i in range(raw_targets.shape[0]):
            sq = raw_targets[i]
            ptype = buffer.board_type[sq[0], sq[1], sq[2]]
            
            if ptype == PieceType.KING:
                # Check for Priest protection
                target_king_color_val = buffer.board_color[sq[0], sq[1], sq[2]]
                
                # Determine priest count from metadata
                # meta[6] = White Priests, meta[7] = Black Priests
                # Map: Color.WHITE(1)->6, Color.BLACK(2)->7
                if target_king_color_val == 1: # White
                     priest_count = buffer.meta[6]
                elif target_king_color_val == 2: # Black
                     priest_count = buffer.meta[7]
                else:
                     priest_count = 0
                     
                if priest_count > 0:
                    continue # King safe due to priests
                
            if ptype == 0:
                # Empty square, usually ignored unless we want to explicit clear
                continue
                
            # If it's a structure (Wall), get all parts
            struct_parts = _get_affected_structure_squares(sq, ptype, buffer)
            
            # Check structure parts for King (unlikely for Wall/Bomb but good practice)
            # Actually Wall implies no King on top.
            
            for part in struct_parts:
                final_targets_list.append(part)
        
        if final_targets_list:
            coords_to_clear = np.unique(np.array(final_targets_list, dtype=COORD_DTYPE), axis=0)
            
    elif is_swap:
        # Swapper Swap: Update both positions
        # Values:
        # To: from_type, from_color
        # From: target_type, target_color
        
        # Validate Wall placement if swapping with Wall?
        # Move generation should catch this, but safe to check.
        
        coords_to_update = np.array([from_coord, to_coord], dtype=COORD_DTYPE)
        new_pieces_data = np.array([
            [target_type, target_color], # At from_coord
            [from_type, from_color]      # At to_coord
        ], dtype=PIECE_TYPE_DTYPE)
        
    else:
        # Standard Move / Capture
        # 1. Handle Capture (Atomic removal at dest)
        if target_type != 0:
             # Capture logic
             # Check if target is structure
             target_structure_squares = _get_affected_structure_squares(to_coord, target_type, buffer)
             
             # If it's a single square (standard), it's overwritten by 'to_coord' update.
             # If it's a Wall, we need to clear the OTHER parts.
             
             if len(target_structure_squares) > 1:
                 # It's a structure. The 'to_coord' will be updated to 'from_piece'.
                 # The OTHER parts must be cleared.
                 others = []
                 for sq in target_structure_squares:
                     if not np.array_equal(sq, to_coord):
                         others.append(sq)
                 
                 if others:
                     coords_to_clear = np.unique(np.array(others, dtype=COORD_DTYPE), axis=0)
        
        # 2. Standard Update
        # From -> Empty
        # To -> Source Piece
        
        # But wait, we separate Clear and Update?
        # OccupancyCache.batch_set_positions can do both (type=0 is clear).
        # MoveEffects distinguishes them for clarity or we just use updates.
        
        # Standard: 
        # From: 0
        # To: from_type, from_color
        # Cleared: (Structure parts)
        
        standard_updates_coords = np.array([from_coord, to_coord], dtype=COORD_DTYPE)
        standard_updates_data = np.array([
            [0, 0],
            [from_type, from_color]
        ], dtype=PIECE_TYPE_DTYPE)
        
        coords_to_update = standard_updates_coords
        new_pieces_data = standard_updates_data
        
        # If we have cleared structure parts, append them as updates (type=0)
        if coords_to_clear.size > 0:
             n_clear = coords_to_clear.shape[0]
             clear_data = np.zeros((n_clear, 2), dtype=PIECE_TYPE_DTYPE)
             coords_to_update = np.vstack([coords_to_update, coords_to_clear])
             new_pieces_data = np.vstack([new_pieces_data, clear_data])
             coords_to_clear = np.empty((0, 3), dtype=COORD_DTYPE) # Consumed
             
    return MoveEffects(coords_to_clear, coords_to_update, new_pieces_data, metadata)

def _get_affected_structure_squares(coord: np.ndarray, piece_type: int, buffer: GameBuffer) -> np.ndarray:
    """Helper to get structure squares using GameBuffer."""
    if piece_type != PieceType.WALL:
        return coord.reshape(1, 3)
    
    # StructureManager now supports dense array (board_type)
    return StructureManager.get_structure_squares_from_component(
        coord, piece_type, buffer.board_type
    )
