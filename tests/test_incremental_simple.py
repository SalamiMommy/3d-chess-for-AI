#!/usr/bin/env python3
"""
Simple test to verify incremental check optimization works correctly.
Compares results of incremental vs slow path to ensure they match.
"""

import sys
import numpy as np

# Test imports
from game3d.game.factory import start_game_state
from game3d.common.shared_types import Color
from game3d.attacks.check import square_attacked_by_incremental, _square_attacked_by_slow

def test_incremental_vs_slow():
    """Test that incremental and slow paths produce same results."""
    print("Testing incremental check detection...")
    print("=" * 70)
    
    # Create initial game state
    state = start_game_state()
    cache = state.cache_manager
    occ_cache = cache.occupancy_cache
    board = state.board
    
    # Get a white piece position
    white_positions = occ_cache.get_positions(Color.WHITE)
    if len(white_positions) == 0:
        print("ERROR: No white pieces found")
        return False
    
    # Use first white piece
    from_coord = white_positions[0]
    to_coord = from_coord.copy()
    to_coord[0] = min(8, to_coord[0] + 1)  # Move one square forward
    
    # Get piece info
    piece_data = occ_cache.get(from_coord)
    if not piece_data:
        print("ERROR: No piece at from_coord")
        return False
    
    # Find king position
    king_pos = occ_cache.find_king(Color.WHITE)
    if king_pos is None:
        print("ERROR: King not found")
        return False
    
    print(f"\nTest setup:")
    print(f"  From: {from_coord} To: {to_coord}")
    print(f"  King at: {king_pos}")
    print(f"  Piece: {piece_data}")
    
    # Simulate the move
    occ_cache.set_position(from_coord, None)
    occ_cache.set_position(to_coord, np.array([piece_data['piece_type'], piece_data['color']]))
    
    try:
        # Test incremental path
        print("\n" + "-" * 70)
        print("Testing INCREMENTAL path...")
        result_incr = square_attacked_by_incremental(
            board, king_pos, Color.BLACK, cache, from_coord, to_coord
        )
        print(f"Result: {result_incr}")
        
        # Test slow path
        print("\nTesting SLOW path...")
        result_slow = _square_attacked_by_slow(
            board, king_pos, Color.BLACK, cache
        )
        print(f"Result: {result_slow}")
        
        # Compare
        print("\n" + "=" * 70)
        if result_incr == result_slow:
            print("✅ SUCCESS: Both methods agree!")
            print(f"   Result: {result_incr}")
            return True
        else:
            print("❌ FAIL: Results differ!")
            print(f"   Incremental: {result_incr}")
            print(f"   Slow: {result_slow}")
            return False
            
    finally:
        # Revert the move
        occ_cache.set_position(to_coord, None)
        occ_cache.set_position(from_coord, np.array([piece_data['piece_type'], piece_data['color']]))

if __name__ == "__main__":
    success = test_incremental_vs_slow()
    sys.exit(0 if success else 1)
