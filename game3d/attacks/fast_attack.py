
"""
Fast attack detection using inverse raycasting and offset checks.
Optimized for Numba.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, SIZE, PieceType, Color,
    PIECE_TYPE_DTYPE, COLOR_DTYPE
)

# =============================================================================
# CONSTANTS & VECTORS - DYNAMIC REGISTRY FROM pieces/pieces/
# =============================================================================

# Import vectors from piece modules
from game3d.pieces.pieces.knight import KNIGHT_MOVEMENT_VECTORS as KNIGHT_VECTORS
from game3d.pieces.pieces.rook import ROOK_MOVEMENT_VECTORS as ROOK_VECTORS
from game3d.pieces.pieces.bishop import BISHOP_MOVEMENT_VECTORS as BISHOP_VECTORS
from game3d.pieces.pieces.queen import QUEEN_MOVEMENT_VECTORS as QUEEN_VECTORS
from game3d.pieces.pieces.trigonalbishop import TRIGONAL_BISHOP_VECTORS as TRIGONAL_VECTORS
from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS as KING_VECTORS
from game3d.pieces.pieces.bigknights import (
    KNIGHT31_MOVEMENT_VECTORS, KNIGHT32_MOVEMENT_VECTORS
)
from game3d.pieces.pieces.spiral import SPIRAL_MOVEMENT_VECTORS
from game3d.pieces.pieces.panel import PANEL_MOVEMENT_VECTORS

# Movement type constants
MOVE_TYPE_JUMP = 0   # Jump pieces - check offset directly
MOVE_TYPE_SLIDER = 1  # Slider pieces - raycast along vectors

# Registry: PieceType.value -> (vectors, movement_type)
PIECE_ATTACK_REGISTRY = {
    # Jump pieces
    PieceType.KNIGHT.value: (KNIGHT_VECTORS, MOVE_TYPE_JUMP),
    PieceType.KING.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.PRIEST.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.KNIGHT31.value: (KNIGHT31_MOVEMENT_VECTORS, MOVE_TYPE_JUMP),
    PieceType.KNIGHT32.value: (KNIGHT32_MOVEMENT_VECTORS, MOVE_TYPE_JUMP),
    PieceType.SPEEDER.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.SLOWER.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.FREEZER.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.HIVE.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.ARMOUR.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.ARCHER.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.BOMB.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.SWAPPER.value: (KING_VECTORS, MOVE_TYPE_JUMP),
    PieceType.PANEL.value: (PANEL_MOVEMENT_VECTORS, MOVE_TYPE_JUMP),
    # Slider pieces
    PieceType.BISHOP.value: (BISHOP_VECTORS, MOVE_TYPE_SLIDER),
    PieceType.ROOK.value: (ROOK_VECTORS, MOVE_TYPE_SLIDER),
    PieceType.QUEEN.value: (QUEEN_VECTORS, MOVE_TYPE_SLIDER),
    PieceType.TRIGONALBISHOP.value: (TRIGONAL_VECTORS, MOVE_TYPE_SLIDER),
    PieceType.SPIRAL.value: (SPIRAL_MOVEMENT_VECTORS, MOVE_TYPE_SLIDER),
}


# =============================================================================
# KERNELS
# =============================================================================

@njit(cache=True, fastmath=True)
def _check_slider_attack(
    target: np.ndarray,
    attacker: np.ndarray,
    occ: np.ndarray,
    vectors: np.ndarray
) -> bool:
    """Check if a slider at 'attacker' hits 'target' using 'vectors'."""
    tx, ty, tz = target
    ax, ay, az = attacker
    
    dx = tx - ax
    dy = ty - ay
    dz = tz - az
    
    # Check alignment with any vector
    # We need to find if (dx, dy, dz) is a multiple of any vector
    # AND if the path is clear.
    
    # Optimization: Check if aligned first
    # This is tricky because vectors are unit-like.
    # Instead of iterating vectors, we can check geometric properties.
    
    # But iterating vectors is robust.
    n_vecs = vectors.shape[0]
    for i in range(n_vecs):
        vx, vy, vz = vectors[i]
        
        # Check if delta is a positive multiple of vector
        # We need k > 0 such that dx = k*vx, dy = k*vy, dz = k*vz
        
        # Check if delta is a positive multiple of vector
        # We need k > 0 such that dx = k*vx, dy = k*vy, dz = k*vz
        
        k = 0
        k_set = False
        valid = True
        
        if vx != 0:
            if dx % vx != 0: valid = False
            else: 
                k = dx // vx
                k_set = True
        elif dx != 0: valid = False
        
        if valid:
            if vy != 0:
                if dy % vy != 0: valid = False
                else:
                    k2 = dy // vy
                    if not k_set: 
                        k = k2
                        k_set = True
                    elif k != k2: valid = False
            elif dy != 0: valid = False
            
        if valid:
            if vz != 0:
                if dz % vz != 0: valid = False
                else:
                    k3 = dz // vz
                    if not k_set:
                        k = k3
                        k_set = True
                    elif k != k3: valid = False
            elif dz != 0: valid = False
            
        if valid and k_set and k > 0:
            # Aligned! Now check path for obstructions.
            # Path is from attacker to target (exclusive of both? NO, exclusive of attacker, inclusive of target?)
            # We are checking if attacker attacks target.
            # So we check squares between attacker and target.
            
            blocked = False
            cx, cy, cz = ax + vx, ay + vy, az + vz
            
            # We iterate k-1 steps
            for _ in range(k - 1):
                if occ[cx, cy, cz] != 0:
                    blocked = True
                    break
                cx += vx
                cy += vy
                cz += vz
                
            if not blocked:
                return True
                
    return False

@njit(cache=True, fastmath=True)
def _fast_attack_kernel(
    target: np.ndarray,
    attacker_coords: np.ndarray,
    attacker_types: np.ndarray,
    occ: np.ndarray,
    attacker_color: int,
    skipped_indices: np.ndarray
) -> int:
    """
    Returns:
        0: Not attacked (by checked pieces)
        1: Attacked
    
    Populates skipped_indices with indices of pieces that were not checked.
    Returns number of skipped pieces in skipped_indices[0] (hacky but works).
    """
    n = attacker_coords.shape[0]
    tx, ty, tz = target
    
    skipped_count = 0
    
    for i in range(n):
        ax, ay, az = attacker_coords[i]
        atype = attacker_types[i]
        
        is_attacking = False
        handled = True
        
        if atype == 1: # PAWN
            # White (1) attacks +Z (z+1), so must be at z-1 relative to target?
            # NO. White pawn at (x,y,z) attacks (x±1, y±1, z+1).
            # So if target is at (tx, ty, tz), a white pawn must be at (tx±1, ty±1, tz-1).
            
            dz = tz - az
            if attacker_color == 1: # White attacker
                if dz == 1:
                    if abs(tx - ax) == 1 and abs(ty - ay) == 1:
                        is_attacking = True
            else: # Black attacker
                if dz == -1:
                    if abs(tx - ax) == 1 and abs(ty - ay) == 1:
                        is_attacking = True
                        
        elif atype == 2: # KNIGHT
            dx = abs(tx - ax)
            dy = abs(ty - ay)
            dz = abs(tz - az)
            # Permutations of (1, 2, 0)
            if (dx == 1 and dy == 2 and dz == 0) or \
               (dx == 2 and dy == 1 and dz == 0) or \
               (dx == 1 and dy == 0 and dz == 2) or \
               (dx == 0 and dy == 1 and dz == 2) or \
               (dx == 2 and dy == 0 and dz == 1) or \
               (dx == 0 and dy == 2 and dz == 1):
                is_attacking = True
                
        elif atype == 3: # BISHOP
            if _check_slider_attack(target, attacker_coords[i], occ, BISHOP_VECTORS):
                is_attacking = True
                
        elif atype == 4: # ROOK
            if _check_slider_attack(target, attacker_coords[i], occ, ROOK_VECTORS):
                is_attacking = True
                
        elif atype == 5: # QUEEN
            if _check_slider_attack(target, attacker_coords[i], occ, QUEEN_VECTORS):
                is_attacking = True
                
        elif atype == 6 or atype == 7: # KING or PRIEST
            # Distance <= 1 in all dims, but not 0
            dx = abs(tx - ax)
            dy = abs(ty - ay)
            dz = abs(tz - az)
            if dx <= 1 and dy <= 1 and dz <= 1 and (dx + dy + dz > 0):
                is_attacking = True
                
        elif atype == 10: # TRIGONAL BISHOP
            if _check_slider_attack(target, attacker_coords[i], occ, TRIGONAL_VECTORS):
                is_attacking = True
                
        else:
            handled = False
            # Add to skipped
            # We use the end of the array to store count? No, passed array is large enough.
            # We'll store count in a separate variable and return it?
            # Numba scalar return is easier.
            # We'll put count in skipped_indices[0] at the end? No, that overwrites data.
            # We'll assume skipped_indices is (N+1,) and store count in [0].
            # Indices start at 1.
            skipped_indices[skipped_count + 1] = i
            skipped_count += 1
            
        if handled and is_attacking:
            return 1 # Attacked!
            
    # Store count
    skipped_indices[0] = skipped_count
    return 0 # Not attacked (yet)

def square_attacked_by_fast(
    board,
    square: np.ndarray,
    attacker_color: int,
    cache
) -> bool:
    """
    Fast check if square is attacked using extended Numba kernel with precomputed data.
    
    ✅ OPTIMIZED: Now delegates to attack_registry.square_attacked_by_extended which:
    - Handles more piece types inline in Numba (20+ vs original 7)
    - Uses precomputed ray data for sliders
    - Minimizes Python loop overhead
    """
    from game3d.attacks.attack_registry import square_attacked_by_extended
    return square_attacked_by_extended(board, square, attacker_color, cache)

