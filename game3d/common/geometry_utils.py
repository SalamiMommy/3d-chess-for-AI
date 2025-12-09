
import numpy as np
from numba import njit
from game3d.common.shared_types import (
    COORD_DTYPE, SIZE, BOOL_DTYPE, PieceType
)

# ✅ OPTIMIZATION: Module-level constants for Numba (extracted from PieceType enum)
# These are used in geometry alignment checks for cache invalidation
_PT_PAWN = PieceType.PAWN.value
_PT_KNIGHT = PieceType.KNIGHT.value
_PT_BISHOP = PieceType.BISHOP.value
_PT_ROOK = PieceType.ROOK.value
_PT_QUEEN = PieceType.QUEEN.value
_PT_KING = PieceType.KING.value
_PT_PRIEST = PieceType.PRIEST.value
_PT_TRIGONAL = PieceType.TRIGONALBISHOP.value
_PT_XYQUEEN = PieceType.XYQUEEN.value
_PT_XZQUEEN = PieceType.XZQUEEN.value
_PT_YZQUEEN = PieceType.YZQUEEN.value
_PT_VECTOR = PieceType.VECTORSLIDER.value
_PT_CONE = PieceType.CONESLIDER.value

@njit(cache=True, nogil=True)
def is_aligned_orthogonal(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Check if points p1 and p2 are aligned orthogonally (share x, y, or z).
    p1: (N, 3) or (3,)
    p2: (M, 3) or (3,)
    Returns boolean array.
    """
    # Optimized for broadcasting if needed, or simple pair check
    # Assume broadcasting p1 (N,3) vs p2 (1,3) most common
    return (p1[:, 0] == p2[:, 0]) | (p1[:, 1] == p2[:, 1]) | (p1[:, 2] == p2[:, 2])

@njit(cache=True, nogil=True)
def is_aligned_diagonal(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Check if points are aligned diagonally (face diagonals).
    |dx|==|dy| or |dy|==|dz| or |dx|==|dz|
    Excluding zeros (orthogonal).
    """
    dx = np.abs(p1[:, 0] - p2[:, 0])
    dy = np.abs(p1[:, 1] - p2[:, 1])
    dz = np.abs(p1[:, 2] - p2[:, 2])
    
    # Check pairs
    xy_match = (dx == dy) & (dx > 0)
    yz_match = (dy == dz) & (dy > 0)
    xz_match = (dx == dz) & (dx > 0)
    
    return xy_match | yz_match | xz_match

@njit(cache=True, nogil=True)
def is_aligned_triagonal(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Check if points are aligned triagonally (space diagonals).
    |dx|==|dy|==|dz|
    """
    dx = np.abs(p1[:, 0] - p2[:, 0])
    dy = np.abs(p1[:, 1] - p2[:, 1])
    dz = np.abs(p1[:, 2] - p2[:, 2])
    
    return (dx == dy) & (dy == dz) & (dx > 0)

@njit(cache=True, nogil=True)
def batch_check_alignment(
    pieces: np.ndarray, 
    affected_coords: np.ndarray, 
    piece_types: np.ndarray
) -> np.ndarray:
    """
    Vectorized check if pieces are geometrically aligned with ANY affected coordinate.
    
    Args:
        pieces: (N, 3) coordinates of candidate pieces
        affected_coords: (M, 3) coordinates of changes
        piece_types: (N,) types of candidate pieces
        
    Returns:
        (N,) boolean mask of pieces to invalidate
    """
    n = pieces.shape[0]
    m = affected_coords.shape[0]
    mask = np.zeros(n, dtype=BOOL_DTYPE)
    
    if n == 0 or m == 0:
        return mask
        
    # We iterate over affected coords (usually small, 1-6)
    # and perform vectorized checks against all pieces (N ~10-50)
    
    # ✅ OPTIMIZATION: Use module-level constants (derived from PieceType enum)
    # Numba resolves these at compile time
    BISHOP = _PT_BISHOP
    ROOK = _PT_ROOK
    QUEEN = _PT_QUEEN
    KING = _PT_KING
    PRIEST = _PT_PRIEST
    TRIGONAL = _PT_TRIGONAL
    XYQUEEN = _PT_XYQUEEN
    XZQUEEN = _PT_XZQUEEN
    YZQUEEN = _PT_YZQUEEN
    VECTOR = _PT_VECTOR
    CONE = _PT_CONE
    PAWN = _PT_PAWN
    KNIGHT = _PT_KNIGHT
    
    for i in range(m):
        target = affected_coords[i:i+1] # Keep dimensions (1, 3) for broadcasting
        
        # 1. Distances
        dx = np.abs(pieces[:, 0] - target[0, 0])
        dy = np.abs(pieces[:, 1] - target[0, 1])
        dz = np.abs(pieces[:, 2] - target[0, 2])
        
        # 2. Alignment Types
        ortho = (dx == 0) | (dy == 0) | (dz == 0) 
        # Refined Ortho: Pure axis alignment?
        # A Rook at (0,0,0) is affected by (5,0,0) [dx!=0, dy=0, dz=0]
        # Yes, sharing ANY coordinate implies plane alignment.
        # Sharing TWO coordinates implies line alignment.
        # Rooks move on LINES.
        line_ortho = ((dx == 0) & (dy == 0)) | ((dx == 0) & (dz == 0)) | ((dy == 0) & (dz == 0))
        
        # Diagonals (Face)
        # |dx|==|dy|!=0 & dz==0, etc.
        # Standard Bishop (3) moves on:
        # (dx=dy, dz=0) z-plane diag
        # (dx=dz, dy=0) y-plane diag
        # (dy=dz, dx=0) x-plane diag
        diag_face = ((dx == dy) & (dz == 0) & (dx > 0)) | \
                    ((dx == dz) & (dy == 0) & (dx > 0)) | \
                    ((dy == dz) & (dx == 0) & (dy > 0))
                    
        # Triagonals (Space)
        triag = (dx == dy) & (dy == dz) & (dx > 0)
        
        # Plane Alignment (Queens/Rooks might control planes)
        # XYQUEEN (17): XY Plane (Ortho + Diag on XY) + King Z
        # Plane alignment means: dz == 0 (XY plane)
        plane_xy = (dz == 0)
        plane_xz = (dy == 0)
        plane_yz = (dx == 0)
        
        # Check by Type
        
        # ROOK (4) - Moves on lines
        mask |= (piece_types == ROOK) & line_ortho
        
        # BISHOP (3) - Moves on face diagonals
        mask |= (piece_types == BISHOP) & diag_face
        
        # TRIGONAL (10) - Moves on space diagonals
        mask |= (piece_types == TRIGONAL) & triag
        
        # QUEEN (5) - Rook + Bishop + Triagonal?
        mask |= (piece_types == QUEEN) & (line_ortho | diag_face | triag)
        
        # PAWN (1) - Distance check (dx=0, dy=0, dz<=2 mostly)
        pawn_prox = (dx <= 1) & (dy <= 1) & (dz <= 2)
        mask |= (piece_types == PAWN) & pawn_prox
        
        # KNIGHT (2) - L-Jumps (1,2,0)
        knight_jump = ((dx==1)&(dy==2)&(dz==0)) | ((dx==2)&(dy==1)&(dz==0)) | \
                      ((dx==1)&(dz==2)&(dy==0)) | ((dx==2)&(dz==1)&(dy==0)) | \
                      ((dy==1)&(dz==2)&(dx==0)) | ((dy==2)&(dz==1)&(dx==0))
        mask |= (piece_types == KNIGHT) & knight_jump
        
        # KING (6) / PRIEST (7) - Adjacent (Rad 1)
        king_prox = (dx <= 1) & (dy <= 1) & (dz <= 1)
        mask |= ((piece_types == KING) | (piece_types == PRIEST)) & king_prox

        # XYQUEEN (17)
        mask |= (piece_types == XYQUEEN) & ((plane_xy & (line_ortho | diag_face)) | ((dx<=1)&(dy<=1)&(dz<=1))) # Plane + King
        
        # SPECIAL PIECES (Always Invalidate for Safety)
        # ECHO (14) - Depends on History (Temporal)
        # NEBULA (13), ORBITER (12), HIVE (11) - Complex/Global effects
        # HIVE: Friendly pieces move together? 
        # For now, mark them as always affected to avoid missing updates.
        mask |= (piece_types == 14) # ECHO
        mask |= (piece_types == 13) # NEBULA
        mask |= (piece_types == 12) # ORBITER
        mask |= (piece_types == 11) # HIVE
        
        # And so on for other variants...
        # Fallback: If type is unknown/complex, assume TRUE to be safe
        is_complex = (piece_types > 21) # Arbitrary threshold for complex/rare pieces
        mask |= is_complex
        
    return mask
