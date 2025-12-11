#!/usr/bin/env python3
"""Reproduce the king position mismatch issue."""

import sys
sys.path.insert(0, '/home/salamimommy/Documents/code/3d-chess-for-AI')

import numpy as np
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.coord_utils import unpack_coords, coords_to_keys

def test_position_tracking():
    """Test if position tracking is consistent after king moves."""
    cache = OccupancyCache()
    
    # Setup: White king at (4,4,0)
    white_king_pos = np.array([4, 4, 0], dtype=COORD_DTYPE)
    cache.set_position_fast(white_king_pos, PieceType.KING.value, Color.WHITE)
    
    # Setup: Black king at (4,3,8)
    black_king_pos = np.array([4, 3, 8], dtype=COORD_DTYPE)
    cache.set_position_fast(black_king_pos, PieceType.KING.value, Color.BLACK)
    
    print("=== Initial State ===")
    print(f"White King (find_king): {cache.find_king(Color.WHITE)}")
    print(f"Black King (find_king): {cache.find_king(Color.BLACK)}")
    
    white_positions = cache.get_positions(Color.WHITE)
    black_positions = cache.get_positions(Color.BLACK)
    print(f"White positions: {white_positions}")
    print(f"Black positions: {black_positions}")
    
    # Now simulate a king move: White king moves from (4,4,0) to (4,3,7)
    # This is the suspicious position
    print("\n=== Move White King from (4,4,0) to (4,3,7) ===")
    
    # Clear old position
    cache.set_position_fast(white_king_pos, 0, 0)
    
    # Set new position  
    new_king_pos = np.array([4, 3, 7], dtype=COORD_DTYPE)
    cache.set_position_fast(new_king_pos, PieceType.KING.value, Color.WHITE)
    
    print(f"White King (find_king): {cache.find_king(Color.WHITE)}")
    print(f"Black King (find_king): {cache.find_king(Color.BLACK)}")
    
    white_positions = cache.get_positions(Color.WHITE)
    black_positions = cache.get_positions(Color.BLACK)
    print(f"White positions: {white_positions}")
    print(f"Black positions: {black_positions}")
    
    # Check what piece type is at each position
    print("\n=== Grid State ===")
    print(f"Piece at (4,4,0): type={cache._ptype[4,4,0]}, color={cache._occ[4,4,0]}")
    print(f"Piece at (4,3,7): type={cache._ptype[4,3,7]}, color={cache._occ[4,3,7]}")
    print(f"Piece at (4,3,8): type={cache._ptype[4,3,8]}, color={cache._occ[4,3,8]}")
    
    # Now move the king again: from (4,3,7) to (4,4,0) (back to original)
    print("\n=== Move White King from (4,3,7) to (4,4,0) (back) ===")
    cache.set_position_fast(new_king_pos, 0, 0)
    cache.set_position_fast(white_king_pos, PieceType.KING.value, Color.WHITE)
    
    print(f"White King (find_king): {cache.find_king(Color.WHITE)}")
    white_positions = cache.get_positions(Color.WHITE)
    print(f"White positions: {white_positions}")
    
    # Check _positions_indices sets
    print("\n=== Position Indices Sets ===")
    white_indices = cache._positions_indices[0]
    black_indices = cache._positions_indices[1]
    print(f"White indices count: {len(white_indices)}")
    print(f"Black indices count: {len(black_indices)}")
    
    # Unpack to see actual coords
    if white_indices:
        white_keys = np.array(list(white_indices), dtype=np.int64)
        white_coords = unpack_coords(white_keys)
        print(f"White coords from indices: {white_coords}")
        
    if black_indices:
        black_keys = np.array(list(black_indices), dtype=np.int64)
        black_coords = unpack_coords(black_keys)
        print(f"Black coords from indices: {black_coords}")
    
    print("\n=== PASS ===" if len(white_indices) == 1 else "\n=== FAIL: Extra indices! ===")

if __name__ == "__main__":
    test_position_tracking()
