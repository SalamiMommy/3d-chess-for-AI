#!/usr/bin/env python3
"""
Advanced Wall Bug Reproduction Script.
Tests wall move generation through the full game pipeline including caching.
"""
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.common.shared_types import PieceType, Color, SIZE, COORD_DTYPE
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.game.gamestate import GameState
from game3d.movement.generator import generate_legal_moves

def test_wall_moves_with_caching():
    """Test wall move generation through the full pipeline with caching."""
    print(f"Testing Wall moves with SIZE={SIZE}")
    print("=" * 60)
    
    # Setup minimal game state
    board = Board()
    cache = OptimizedCacheManager(board)
    
    # Place a wall at various positions and check generated moves
    test_positions = [
        [5, 7, 4],   # Valid anchor (occupies y=7,8)
        [5, 6, 4],   # Valid anchor (occupies y=6,7)
        [0, 0, 0],   # Corner
        [7, 7, 4],   # x=7,y=7 (occupies up to x=8,y=8)
    ]
    
    for wall_anchor in test_positions:
        print(f"\n--- Testing Wall at {wall_anchor} ---")
        
        # Clear board
        cache.occupancy_cache._occ[:] = 0
        cache.occupancy_cache._ptype[:] = 0
        
        # Place wall (all 4 squares)
        anchor = np.array(wall_anchor, dtype=COORD_DTYPE)
        offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=COORD_DTYPE)
        wall_squares = anchor + offsets
        
        # Check if all wall squares are in bounds
        valid = True
        for sq in wall_squares:
            if not (0 <= sq[0] < SIZE and 0 <= sq[1] < SIZE and 0 <= sq[2] < SIZE):
                print(f"  INVALID: Wall square {sq} is out of bounds!")
                valid = False
                break
        
        if not valid:
            continue
            
        # Place the wall
        for sq in wall_squares:
            cache.occupancy_cache.set_position(
                sq,
                {"piece_type": PieceType.WALL, "color": Color.WHITE}
            )
        
        # Create game state
        state = GameState(board=board, cache_manager=cache, color=Color.WHITE)
        
        # Generate legal moves
        try:
            legal_moves = generate_legal_moves(state)
            print(f"  Generated {len(legal_moves)} legal moves")
            
            # Check for invalid moves
            invalid_count = 0
            for move in legal_moves:
                from_pos = move[:3]
                to_pos = move[3:6]
                
                # Check if this is a wall move
                piece = cache.occupancy_cache.get(from_pos)
                if piece and piece["piece_type"] == PieceType.WALL:
                    # Check if destination is valid for wall
                    if to_pos[0] >= SIZE - 1 or to_pos[1] >= SIZE - 1:
                        print(f"  ❌ INVALID MOVE FOUND: {from_pos} -> {to_pos}")
                        print(f"     Destination would extend to [{to_pos[0]+1}, {to_pos[1]+1}, {to_pos[2]}]")
                        invalid_count += 1
            
            if invalid_count == 0:
                print(f"  ✓ All {len(legal_moves)} moves are valid")
            else:
                print(f"  ✗ Found {invalid_count} invalid moves!")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

def test_wall_specific_scenario():
    """Test the specific scenario from the error message."""
    print("\n" + "=" * 60)
    print("Testing Specific Scenario: Wall trying to move to [5, 8, 4]")
    print("=" * 60)
    
    # Setup
    board = Board()
    cache = OptimizedCacheManager(board)
    
    # Try different source positions that might generate a move to [5, 8, 4]
    # The wall could be at:
    # - [5, 7, 4] trying to move +1 in Y direction -> [5, 8, 4] ✗ (should be blocked)
    # - [4, 8, 4] trying to move +1 in X direction -> [5, 8, 4] ✗ (should be blocked)
    # - [5, 8, 3] trying to move +1 in Z direction -> [5, 8, 4] ✗ (should be blocked)
    # - [5, 8, 5] trying to move -1 in Z direction -> [5, 8, 4] ✗ (should be blocked)
    
    # But first, [5, 8, 4] itself is INVALID as a wall anchor because it extends to y=9
    
    source_positions = [
        [5, 7, 4],  # Moving from y=7 to y=8
        [4, 8, 4],  # Moving from x=4 to x=5 (but y=8 is still invalid)
    ]
    
    for src in source_positions:
        print(f"\n--- Wall at {src} ---")
        
        # Clear board
        cache.occupancy_cache._occ[:] = 0
        cache.occupancy_cache._ptype[:] = 0
        
        # Place wall
        anchor = np.array(src, dtype=COORD_DTYPE)
        offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=COORD_DTYPE)
        wall_squares = anchor + offsets
        
        # Check validity
        valid = all(0 <= sq[0] < SIZE and 0 <= sq[1] < SIZE and 0 <= sq[2] < SIZE 
                   for sq in wall_squares)
        
        if not valid:
            print(f"  Source position {src} is INVALID for a wall (extends out of bounds)")
            continue
        
        # Place wall
        for sq in wall_squares:
            cache.occupancy_cache.set_position(
                sq,
                {"piece_type": PieceType.WALL, "color": Color.WHITE}
            )
        
        # Generate moves
        state = GameState(board=board, cache_manager=cache, color=Color.WHITE)
        legal_moves = generate_legal_moves(state)
        
        # Check for move to [5, 8, 4]
        target = np.array([5, 8, 4])
        found = False
        
        for move in legal_moves:
            if np.array_equal(move[3:6], target):
                print(f"  ❌ FOUND INVALID MOVE: {move[:3]} -> {move[3:6]}")
                found = True
                
        if not found:
            print(f"  ✓ No move to [5, 8, 4] generated (correct)")

if __name__ == "__main__":
    test_wall_moves_with_caching()
    test_wall_specific_scenario()
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
