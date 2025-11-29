"""Pawn movement generator - fully numpy native with vectorized operations."""
import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING

from game3d.common.coord_utils import in_bounds_vectorized
from game3d.common.shared_types import (
    Color, PieceType,
    COORD_DTYPE, COLOR_WHITE, COLOR_BLACK, SIZE,
    PAWN_START_RANK_WHITE, PAWN_START_RANK_BLACK,
    PAWN_PROMOTION_RANK_WHITE, PAWN_PROMOTION_RANK_BLACK, MOVE_FLAGS,
    COLOR_DTYPE, PIECE_TYPE_DTYPE
)
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.game.gamestate import GameState

# Pawn push directions - white moves +Y, black moves -Y
PAWN_PUSH_DIRECTIONS = np.array([
    [0, 1, 0],   # White pawn push
    [0, -1, 0],  # Black pawn push
], dtype=COORD_DTYPE)

# Pawn attack directions - 4 trigonal attacks
# White (+Y): (±1, 1, ±1)
# Black (-Y): (±1, -1, ±1)
PAWN_ATTACK_DIRECTIONS = np.array([
    [1, 1, 1], [-1, 1, 1], [1, 1, -1], [-1, 1, -1],  # White attacks
    [1, -1, 1], [-1, -1, 1], [1, -1, -1], [-1, -1, -1],  # Black attacks
], dtype=COORD_DTYPE)

# ✅ OPTIMIZATION: Pre-compute attack direction slices
PAWN_ATTACK_DIRECTIONS_WHITE = PAWN_ATTACK_DIRECTIONS[:4]
PAWN_ATTACK_DIRECTIONS_BLACK = PAWN_ATTACK_DIRECTIONS[4:]


@njit(cache=True, fastmath=True)
def _filter_pawn_captures(
    targets: np.ndarray,
    occ: np.ndarray,
    ptype: np.ndarray,
    player_color: int,
    armour_type: int
) -> np.ndarray:
    """Fused capture filtering: occupancy + type check in one pass.
    
    OPTIMIZATION: Eliminates intermediate arrays (dest_colors, dest_types)
    by combining lookup and filtering in a single pass. Better cache locality.
    """
    n = targets.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
    
    mask = np.empty(n, dtype=np.bool_)
    
    # Single pass: lookup color + type, apply combined filter
    for i in range(n):
        x, y, z = targets[i, 0], targets[i, 1], targets[i, 2]
        target_color = occ[x, y, z]
        target_type = ptype[x, y, z]
        
        # Combined check: enemy piece that is not armour
        mask[i] = (
            (target_color != 0) & 
            (target_color != player_color) & 
            (target_type != armour_type)
        )
    
    return targets[mask]


def _is_armoured(piece) -> bool:
    """Check if piece has armour protection."""
    return piece is not None and (
        (hasattr(piece, "armoured") and piece.armoured) or
        piece["piece_type"] == PieceType.ARMOUR
    )

def _is_on_start_rank(y: int, colour: Color) -> bool:
    """Check if pawn is on starting rank (using Y coordinate)."""
    return (colour == Color.WHITE and y == PAWN_START_RANK_WHITE) or \
           (colour == Color.BLACK and y == PAWN_START_RANK_BLACK)

def _is_promotion_rank(y: int, colour: Color) -> bool:
    """Check if pawn is on promotion rank (using Y coordinate)."""
    return (colour == Color.WHITE and y == PAWN_PROMOTION_RANK_WHITE) or \
           (colour == Color.BLACK and y == PAWN_PROMOTION_RANK_BLACK)

def generate_pawn_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate pawn moves with optimized batch processing.
    
    Supports both single coordinate (3,) and batch coordinates (N, 3).
    """
    # Normalize input to (N, 3)
    if pos.ndim == 1:
        coords = pos.reshape(1, 3)
    else:
        coords = pos
        
    if coords.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)
        
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    colour = Color(color)

    move_arrays = []
    
    # Select appropriate push direction based on color
    dy = 1 if colour == Color.WHITE else -1
    start_rank = PAWN_START_RANK_WHITE if colour == Color.WHITE else PAWN_START_RANK_BLACK
    
    # --- 1. PUSH MOVES ---
    
    # Single push targets
    push_y = y + dy
    
    # Double push targets
    two_step_y = y + 2 * dy
    
    # Filter valid single pushes (in bounds)
    # x and z are same, only y changes
    valid_push_mask = (push_y >= 0) & (push_y < SIZE)
    
    if np.any(valid_push_mask):
        # Check occupancy for single push
        # Construct target coordinates for valid pushes
        push_targets = np.empty((np.sum(valid_push_mask), 3), dtype=COORD_DTYPE)
        push_targets[:, 0] = x[valid_push_mask]
        push_targets[:, 1] = push_y[valid_push_mask]
        push_targets[:, 2] = z[valid_push_mask]
        
        # Batch lookup occupancy (colors only)
        # Use unsafe variant as we checked bounds
        push_colors = cache_manager.occupancy_cache.batch_get_colors_only(push_targets)
        
        # Empty squares allow push
        empty_push_mask = (push_colors == 0)
        
        if np.any(empty_push_mask):
            # Add single push moves
            n_pushes = np.sum(empty_push_mask)
            push_moves = np.empty((n_pushes, 6), dtype=COORD_DTYPE)
            
            # Source coords (subset of valid pushes that are also empty)
            # We need to map back to original indices or just use the subset
            # valid_push_mask selects from original coords
            # empty_push_mask selects from push_targets
            
            # Get indices of original coords that have valid AND empty push
            # This is a bit tricky with boolean indexing.
            # Let's use integer indexing for clarity.
            valid_indices = np.where(valid_push_mask)[0]
            successful_push_indices = valid_indices[empty_push_mask]
            
            push_moves[:, 0] = x[successful_push_indices]
            push_moves[:, 1] = y[successful_push_indices]
            push_moves[:, 2] = z[successful_push_indices]
            push_moves[:, 3] = x[successful_push_indices]
            push_moves[:, 4] = push_y[successful_push_indices]
            push_moves[:, 5] = z[successful_push_indices]
            
            move_arrays.append(push_moves)
            
            # --- 2. DOUBLE PUSH MOVES ---
            # Can only double push if:
            # 1. On start rank
            # 2. Single push was valid AND empty (we already filtered for this)
            # 3. Double push target is in bounds
            # 4. Double push target is empty
            
            # Check start rank for successful pushes
            on_start_rank = (y[successful_push_indices] == start_rank)
            
            # Check double push bounds
            double_target_y = two_step_y[successful_push_indices]
            in_bounds_double = (double_target_y >= 0) & (double_target_y < SIZE)
            
            # Candidates for double push
            double_candidates_mask = on_start_rank & in_bounds_double
            
            if np.any(double_candidates_mask):
                candidate_indices = successful_push_indices[double_candidates_mask]
                
                # Construct double push targets
                double_targets = np.empty((len(candidate_indices), 3), dtype=COORD_DTYPE)
                double_targets[:, 0] = x[candidate_indices]
                double_targets[:, 1] = two_step_y[candidate_indices] # Use precomputed y
                double_targets[:, 2] = z[candidate_indices]
                
                # Check occupancy
                double_colors = cache_manager.occupancy_cache.batch_get_colors_only(double_targets)
                empty_double_mask = (double_colors == 0)
                
                if np.any(empty_double_mask):
                    n_double = np.sum(empty_double_mask)
                    double_moves = np.empty((n_double, 6), dtype=COORD_DTYPE)
                    
                    final_indices = candidate_indices[empty_double_mask]
                    
                    double_moves[:, 0] = x[final_indices]
                    double_moves[:, 1] = y[final_indices]
                    double_moves[:, 2] = z[final_indices]
                    double_moves[:, 3] = x[final_indices]
                    double_moves[:, 4] = two_step_y[final_indices]
                    double_moves[:, 5] = z[final_indices]
                    
                    move_arrays.append(double_moves)

    # --- 3. CAPTURE MOVES ---
    
    # Attack directions
    attack_dirs = PAWN_ATTACK_DIRECTIONS_WHITE if color == COLOR_WHITE else PAWN_ATTACK_DIRECTIONS_BLACK
    # Shape (4, 3)
    
    # We need to generate all potential captures for all pawns
    # (N, 1, 3) + (1, 4, 3) -> (N, 4, 3)
    potential_captures = coords[:, np.newaxis, :] + attack_dirs[np.newaxis, :, :]
    
    # Flatten for batch processing
    flat_captures = potential_captures.reshape(-1, 3)
    
    # Filter out of bounds
    in_bounds_mask = (
        (flat_captures[:, 0] >= 0) & (flat_captures[:, 0] < SIZE) &
        (flat_captures[:, 1] >= 0) & (flat_captures[:, 1] < SIZE) &
        (flat_captures[:, 2] >= 0) & (flat_captures[:, 2] < SIZE)
    )
    
    valid_captures = flat_captures[in_bounds_mask]
    
    if valid_captures.shape[0] > 0:
        # Check occupancy and type
        # Use fused filter
        final_dests = _filter_pawn_captures(
            valid_captures,
            cache_manager.occupancy_cache._occ,
            cache_manager.occupancy_cache._ptype,
            color,
            PieceType.ARMOUR.value
        )
        
        if final_dests.shape[0] > 0:
            # We have valid capture destinations.
            # We need to reconstruct the moves (from -> to).
            # Since we flattened, we lost the mapping.
            # But we can recover 'from' because capture is always fixed offset?
            # No, 'from' = 'to' - 'offset'. But we have 4 offsets.
            # Better to keep track of indices.
            
            # Alternative: Don't flatten immediately, or replicate 'from' coords.
            
            # Let's replicate 'from' coords to match flat_captures
            # coords: (N, 3) -> repeat 4 times -> (N, 4, 3) -> flatten -> (N*4, 3)
            flat_starts = np.repeat(coords, 4, axis=0)
            
            # Apply same masks
            valid_starts = flat_starts[in_bounds_mask]
            
            # Now we need to apply the same filter as _filter_pawn_captures but keep pairs
            # _filter_pawn_captures returns only destinations.
            # Let's inline the filter logic here to keep pairs.
            
            occ = cache_manager.occupancy_cache._occ
            ptype = cache_manager.occupancy_cache._ptype
            armour_type = PieceType.ARMOUR.value
            
            # Vectorized lookup
            cx, cy, cz = valid_captures[:, 0], valid_captures[:, 1], valid_captures[:, 2]
            target_colors = occ[cx, cy, cz]
            target_types = ptype[cx, cy, cz]
            
            capture_mask = (
                (target_colors != 0) & 
                (target_colors != color) & 
                (target_types != armour_type)
            )
            
            if np.any(capture_mask):
                n_caps = np.sum(capture_mask)
                cap_moves = np.empty((n_caps, 6), dtype=COORD_DTYPE)
                
                cap_moves[:, :3] = valid_starts[capture_mask]
                cap_moves[:, 3:] = valid_captures[capture_mask]
                
                move_arrays.append(cap_moves)

    if not move_arrays:
        return np.empty((0, 6), dtype=COORD_DTYPE)
    
    return np.concatenate(move_arrays)

@register(PieceType.PAWN)
def pawn_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_pawn_moves(state.cache_manager, state.color, pos)

__all__ = ['generate_pawn_moves', 'PAWN_PUSH_DIRECTIONS', 'PAWN_ATTACK_DIRECTIONS']
