"""Pawn movement generator - fully numpy native with vectorized operations."""
import numpy as np
from numba import njit
from typing import List, TYPE_CHECKING

from game3d.common.coord_utils import in_bounds_vectorized
from game3d.common.shared_types import *

if TYPE_CHECKING: pass

# Pawn push directions - white moves +Z, black moves -Z
PAWN_PUSH_DIRECTIONS = np.array([
    [0, 0, 1],
    [0, 0, -1],
], dtype=COORD_DTYPE)

# Pawn attack directions - 4 trigonal attacks (forward in Z, diagonal in XY)
# White (+Z): (±1, ±1, 1)
# Black (-Z): (±1, ±1, -1)
PAWN_ATTACK_DIRECTIONS = np.array([
    [-1, -1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, -1, -1],
    [-1, 1, -1],
    [1, -1, -1],
    [1, 1, -1],
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
    dz: int,
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
        pz = z + dz
        if 0 <= pz < SIZE:
            if occ[x, y, pz] == 0:
                # Add push
                moves[count, 0] = x
                moves[count, 1] = y
                moves[count, 2] = z
                moves[count, 3] = x
                moves[count, 4] = y
                moves[count, 5] = pz
                count += 1
                
                # 2. Double Push (only if single push valid and empty)
                if z == start_rank:
                    ppz = z + 2 * dz
                    if 0 <= ppz < SIZE:
                        if occ[x, y, ppz] == 0:
                            moves[count, 0] = x
                            moves[count, 1] = y
                            moves[count, 2] = z
                            moves[count, 3] = x
                            moves[count, 4] = y
                            moves[count, 5] = ppz
                            count += 1
                            
        # 3. Captures
        for j in range(4):
            dx, dy, dz_attack = attack_dirs[j]
            tx, ty, tz = x + dx, y + dy, z + dz_attack
            
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

__all__ = ['PAWN_PUSH_DIRECTIONS', 'PAWN_ATTACK_DIRECTIONS']

