"""Pin Detection Logic.

Identifies pieces that are pinned to the King.
A piece is pinned if moving it would expose the King to check.

Algorithm:
1. Identify enemy sliders that are geometrically aligned with the King (using raw moves).
2. Trace the ray from the enemy slider to the King.
3. Count pieces on the ray.
4. If exactly one friendly piece is on the ray, it is pinned.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from numba import njit

from game3d.common.shared_types import (
    COORD_DTYPE, SIZE, Color, PieceType
)
from game3d.common.coord_utils import in_bounds_vectorized
from game3d.movement.pseudolegal import coord_to_key
from game3d.game.gamestate import GameState

@njit(cache=True)
def _get_ray_between(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Returns all coordinates on the line segment between start and end (exclusive).
    Returns empty array if not aligned.
    """
    diff = end - start
    dist = np.max(np.abs(diff))
    
    if dist == 0:
        return np.empty((0, 3), dtype=COORD_DTYPE)
        
    step = diff // dist
    
    # Check if it's a valid straight line (diagonal or orthogonal)
    # For a valid move, the absolute values of non-zero components of step must be 1
    # and the diff must be a multiple of step.
    
    # Simple check: cross product for 3D alignment is complex, 
    # but for chess moves (ortho/diag), step components are -1, 0, 1.
    # And start + k * step == end for some k.
    
    # Verify alignment
    if not np.all(start + step * dist == end):
        return np.empty((0, 3), dtype=COORD_DTYPE)
        
    # Generate ray
    ray = np.empty((dist - 1, 3), dtype=COORD_DTYPE)
    curr = start.copy()
    
    for i in range(dist - 1):
        curr += step
        ray[i] = curr
        
    return ray

def get_pinned_pieces(state: GameState, color: int) -> Dict[int, int]:
    """
    Identify pinned pieces for the given color.
    
    Returns:
        Dict[piece_key, pin_direction_key]
        - piece_key: Coordinate key of the pinned piece
        - pin_direction_key: Coordinate key of the pinning direction (step vector)
                             OR some representation of the allowed movement line.
                             Actually, usually we return the allowed movement mask or ray.
                             For simplicity, let's return the set of allowed squares (ray + attacker pos).
                             But for now, let's just return the attacker's position key, 
                             as the piece can only move towards the attacker or capture it.
                             Wait, it can also move AWAY from attacker if it stays on the line? 
                             No, usually it can only move ON the line segment between King and Attacker.
    """
    pinned = {} # key -> attacker_key (or ray mask?)
    
    # 1. Find King
    king_pos = state.cache_manager.occupancy_cache.find_king(color)
    if king_pos is None:
        return {}
        
    opponent_color = Color(color).opposite()
    
    # 2. Get Enemy Raw Moves (Geometric attacks)
    # We need raw moves to see through pieces
    enemy_raw_moves = state.cache_manager.move_cache.get_raw_moves(opponent_color)
    
    if enemy_raw_moves is None or enemy_raw_moves.size == 0:
        return {}
        
    # 3. Filter for moves that hit the King
    # enemy_raw_moves: [from_x, from_y, from_z, to_x, to_y, to_z]
    
    # Mask for moves targeting King
    hits_king_mask = (enemy_raw_moves[:, 3] == king_pos[0]) & \
                     (enemy_raw_moves[:, 4] == king_pos[1]) & \
                     (enemy_raw_moves[:, 5] == king_pos[2])
                     
    attacking_moves = enemy_raw_moves[hits_king_mask]
    
    if attacking_moves.size == 0:
        return {}
        
    # 4. Analyze each potential pin
    occ_cache = state.cache_manager.occupancy_cache
    
    for move in attacking_moves:
        attacker_pos = move[:3]
        
        # Get ray between attacker and King
        ray = _get_ray_between(attacker_pos, king_pos)
        
        if ray.size == 0:
            continue
            
        # Check occupancy on the ray
        # We need to count pieces on this ray
        
        blockers = []
        for i in range(ray.shape[0]):
            sq = ray[i]
            piece = occ_cache.get(sq)
            if piece:
                blockers.append((sq, piece))
                
        # Pin condition: Exactly one friendly piece and NO enemy pieces
        # (If there's an enemy piece, it blocks the check itself, so no pin on friendly)
        
        friendly_blockers = [b for b in blockers if b[1]['color'] == color]
        enemy_blockers = [b for b in blockers if b[1]['color'] == opponent_color]
        
        if len(friendly_blockers) == 1 and len(enemy_blockers) == 0:
            # We have a pin!
            pinned_sq, _ = friendly_blockers[0]
            pinned_key = int(coord_to_key(pinned_sq.reshape(1, 3))[0])
            attacker_key = int(coord_to_key(attacker_pos.reshape(1, 3))[0])
            
            # Store pin info
            # We store the attacker key. The pinned piece can only move to squares 
            # that mask the check (which is the ray + attacker pos).
            # But for now, just identifying it is enough for the `is_pinned` check.
            pinned[pinned_key] = attacker_key
            
    return pinned

def get_legal_pin_squares(king_pos: np.ndarray, attacker_pos: np.ndarray) -> Set[int]:
    """
    Get the set of legal coordinate keys for a pinned piece.
    A pinned piece can only move along the ray between the King and the attacker,
    or capture the attacker.
    
    Returns:
        Set of coordinate keys (int) representing allowed squares.
    """
    allowed_keys = set()
    
    # 1. Add attacker position (capture)
    attacker_key = int(coord_to_key(attacker_pos.reshape(1, 3))[0])
    allowed_keys.add(attacker_key)
    
    # 2. Add ray squares (block)
    ray = _get_ray_between(attacker_pos, king_pos)
    if ray.size > 0:
        ray_keys = coord_to_key(ray)
        for k in ray_keys:
            allowed_keys.add(int(k))
            
    return allowed_keys

__all__ = ['get_pinned_pieces', 'get_legal_pin_squares']
