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
def _generate_pawn_moves_batch_kernel(
    positions: np.ndarray,
    occ: np.ndarray,
    ptype: np.ndarray,
    color: int,
    start_rank: int,
    dy: int,
    attack_dirs: np.ndarray,
    armour_type: int
) -> np.ndarray:
    """Fused kernel for generating all pawn moves (push, double push, captures)."""
    n = positions.shape[0]
    # Max moves per pawn: 1 push + 1 double push + 4 captures = 6
    max_moves = n * 6
    moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    count = 0
    
    for i in range(n):
        x, y, z = positions[i]
        
        # 1. Single Push
        py = y + dy
        if 0 <= py < SIZE:
            if occ[x, py, z] == 0:
                # Add push
                moves[count, 0] = x
                moves[count, 1] = y
                moves[count, 2] = z
                moves[count, 3] = x
                moves[count, 4] = py
                moves[count, 5] = z
                count += 1
                
                # 2. Double Push (only if single push valid and empty)
                if y == start_rank:
                    ppy = y + 2 * dy
                    if 0 <= ppy < SIZE:
                        if occ[x, ppy, z] == 0:
                            moves[count, 0] = x
                            moves[count, 1] = y
                            moves[count, 2] = z
                            moves[count, 3] = x
                            moves[count, 4] = ppy
                            moves[count, 5] = z
                            count += 1
                            
        # 3. Captures
        for j in range(4):
            dx, dy_attack, dz = attack_dirs[j]
            tx, ty, tz = x + dx, y + dy_attack, z + dz
            
            if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                target_color = occ[tx, ty, tz]
                target_type = ptype[tx, ty, tz]
                
                # Capture enemy piece that is not armour
                if target_color != 0 and target_color != color and target_type != armour_type:
                    moves[count, 0] = x
                    moves[count, 1] = y
                    moves[count, 2] = z
                    moves[count, 3] = tx
                    moves[count, 4] = ty
                    moves[count, 5] = tz
                    count += 1
                    
    return moves[:count]


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
        
    colour = Color(color)
    
    # Select appropriate parameters
    dy = 1 if colour == Color.WHITE else -1
    start_rank = PAWN_START_RANK_WHITE if colour == Color.WHITE else PAWN_START_RANK_BLACK
    attack_dirs = PAWN_ATTACK_DIRECTIONS_WHITE if colour == Color.WHITE else PAWN_ATTACK_DIRECTIONS_BLACK
    
    # Get cache arrays
    occ = cache_manager.occupancy_cache._occ
    ptype = cache_manager.occupancy_cache._ptype
    armour_type = PieceType.ARMOUR.value
    
    # Use fused kernel
    return _generate_pawn_moves_batch_kernel(
        coords,
        occ,
        ptype,
        color,
        start_rank,
        dy,
        attack_dirs,
        armour_type
    )

@register(PieceType.PAWN)
def pawn_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    return generate_pawn_moves(state.cache_manager, state.color, pos)

__all__ = ['generate_pawn_moves', 'PAWN_PUSH_DIRECTIONS', 'PAWN_ATTACK_DIRECTIONS']
