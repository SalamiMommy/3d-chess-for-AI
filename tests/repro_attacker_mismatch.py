#!/usr/bin/env python3
"""Reproduce the king attacker position mismatch issue.

Log shows:
- White King at (2,6,1)
- Black King at (3,7,3)
- Attacker: KING at (2,6,2)

The attacker position doesn't match either king!
"""

import sys
sys.path.insert(0, '/home/salamimommy/Documents/code/3d-chess-for-AI')

import numpy as np
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.coord_utils import unpack_coords, coords_to_keys

def test_attacker_mismatch():
    """Simulate the scenario from the log and check for mismatches."""
    cache = OccupancyCache()
    
    # Final positions from log:
    white_king_pos = np.array([2, 6, 1], dtype=COORD_DTYPE)  # White King at (2,6,1)
    black_king_pos = np.array([3, 7, 3], dtype=COORD_DTYPE)  # Black King at (3,7,3)
    
    # Setup initial state - simulate king moving from (2,6,2) to (2,6,1)
    initial_white_king = np.array([2, 6, 2], dtype=COORD_DTYPE)  
    
    # Set initial positions
    cache.set_position_fast(initial_white_king, PieceType.KING.value, Color.WHITE)
    cache.set_position_fast(black_king_pos, PieceType.KING.value, Color.BLACK)
    
    print("=== Initial State ===")
    print(f"White King (find_king): {cache.find_king(Color.WHITE)}")
    print(f"Black King (find_king): {cache.find_king(Color.BLACK)}")
    
    white_positions = cache.get_positions(Color.WHITE)
    print(f"White positions (get_positions): {white_positions}")
    
    # Now move whithe king from (2,6,2) to (2,6,1)
    print("\n=== Move White King from (2,6,2) to (2,6,1) ===")
    cache.set_position_fast(initial_white_king, 0, 0)  # Clear old position
    cache.set_position_fast(white_king_pos, PieceType.KING.value, Color.WHITE)  # Set new position
    
    print(f"White King (find_king): {cache.find_king(Color.WHITE)}")
    print(f"Black King (find_king): {cache.find_king(Color.BLACK)}")
    
    white_positions = cache.get_positions(Color.WHITE)
    print(f"White positions (get_positions): {white_positions}")
    
    # Verify grid state
    print("\n=== Grid State ===")
    print(f"Piece at (2,6,1): type={cache._ptype[2,6,1]}, color={cache._occ[2,6,1]}")
    print(f"Piece at (2,6,2): type={cache._ptype[2,6,2]}, color={cache._occ[2,6,2]}")
    print(f"Piece at (3,7,3): type={cache._ptype[3,7,3]}, color={cache._occ[3,7,3]}")
    
    # Check the positions_indices sets
    print("\n=== Position Indices Sets ===")
    white_indices = cache._positions_indices[0]
    print(f"White indices: {white_indices}")
    
    if white_indices:
        white_keys = np.array(list(white_indices), dtype=np.int64)
        white_coords = unpack_coords(white_keys)
        print(f"White coords from indices: {white_coords}")
    
    # Verify consistency
    find_king_pos = cache.find_king(Color.WHITE)
    get_pos_result = cache.get_positions(Color.WHITE)
    
    if len(get_pos_result) != 1:
        print(f"\n=== FAIL: Expected 1 white position, got {len(get_pos_result)} ===")
        return False
        
    if not np.array_equal(find_king_pos, get_pos_result[0]):
        print(f"\n=== FAIL: find_king {find_king_pos} != get_positions {get_pos_result[0]} ===")
        return False
    
    if not np.array_equal(find_king_pos, white_king_pos):
        print(f"\n=== FAIL: King position {find_king_pos} != expected {white_king_pos} ===")
        return False
    
    # Check that old position is really cleared
    old_ptype = cache._ptype[2, 6, 2]
    old_color = cache._occ[2, 6, 2]
    if old_ptype != 0 or old_color != 0:
        print(f"\n=== FAIL: Old position (2,6,2) not cleared: type={old_ptype}, color={old_color} ===")
        return False
    
    print("\n=== PASS ===")
    return True

def test_batch_update():
    """Test batch_set_positions also maintains consistency."""
    cache = OccupancyCache()
    
    # Use batch_set_positions to setup initial state
    coords = np.array([
        [2, 6, 2],  # White King initial
        [3, 7, 3],  # Black King
    ], dtype=COORD_DTYPE)
    
    pieces = np.array([
        [PieceType.KING.value, Color.WHITE],
        [PieceType.KING.value, Color.BLACK],
    ], dtype=np.int8)
    
    cache.batch_set_positions(coords, pieces)
    
    print("\n=== Test batch_set_positions ===")
    print(f"White King (find_king): {cache.find_king(Color.WHITE)}")
    print(f"White positions: {cache.get_positions(Color.WHITE)}")
    
    # Move white king using batch_set_positions
    move_coords = np.array([
        [2, 6, 2],  # Clear old position  
        [2, 6, 1],  # New position
    ], dtype=COORD_DTYPE)
    
    move_pieces = np.array([
        [0, 0],  # Empty
        [PieceType.KING.value, Color.WHITE],  # King at new position
    ], dtype=np.int8)
    
    cache.batch_set_positions(move_coords, move_pieces)
    
    print(f"After move - White King (find_king): {cache.find_king(Color.WHITE)}")
    print(f"After move - White positions: {cache.get_positions(Color.WHITE)}")
    
    find_king_pos = cache.find_king(Color.WHITE)
    expected = np.array([2, 6, 1], dtype=COORD_DTYPE)
    
    if not np.array_equal(find_king_pos, expected):
        print(f"\n=== BATCH FAIL: King position {find_king_pos} != expected {expected} ===")
        return False
    
    print("=== BATCH PASS ===")
    return True

if __name__ == "__main__":
    ok1 = test_attacker_mismatch()
    ok2 = test_batch_update()
    
    if ok1 and ok2:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")
        sys.exit(1)
