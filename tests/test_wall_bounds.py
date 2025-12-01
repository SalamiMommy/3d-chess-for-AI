#!/usr/bin/env python3
"""Quick test to verify Wall bounds checking fix."""

import numpy as np
import sys
sys.path.insert(0, '/home/salamimommy/Documents/code/3d-chess-for-AI')

from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE
from game3d.cache.manager import OptimizedCacheManager
from game3d.pieces.pieces.wall import generate_wall_moves

def test_wall_edge_moves():
    """Test that Wall moves near board edges don't cause IndexError."""
    print(f"Testing Wall movement on {SIZE}x{SIZE}x{SIZE} board...")
    
    # Create cache manager
    cache_manager = OptimizedCacheManager()
    
    # Test case 1: Wall at edge position (8, 8, 4) should not crash
    print("\nTest 1: Wall at position (8, 8, 4)")
    try:
        # Place a wall piece (anchor at 8, 8, 4 is INVALID since it needs room for 2x2 block)
        # Valid anchor positions are at most (7, 7, z)
        wall_pos = np.array([8, 8, 4], dtype=COORD_DTYPE)
        
        # Try to place wall at each of the 4 squares
        for offset in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]:
            pos = wall_pos + np.array(offset, dtype=COORD_DTYPE)
            # Check bounds first
            if np.all(pos < SIZE) and np.all(pos >= 0):
                cache_manager.occupancy_cache.set_position(
                    pos,
                    np.array([PieceType.WALL.value, Color.WHITE], dtype=np.int8)
                )
        
        # Generate moves - should not crash
        moves = generate_wall_moves(cache_manager, Color.WHITE, wall_pos)
        print(f"  Generated {len(moves)} moves (expected 0 or very few due to bounds)")
        print(f"  ✓ No crash!")
        
    except IndexError as e:
        print(f"  ✗ FAILED with IndexError: {e}")
        return False
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        return False
    
    # Clear board
    cache_manager.occupancy_cache.clear()
    
    # Test case 2: Wall at valid edge position (7, 7, 4)
    print("\nTest 2: Wall at position (7, 7, 4)")
    try:
        wall_pos = np.array([7, 7, 4], dtype=COORD_DTYPE)
        
        # Place wall at the 4 squares
        for offset in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]:
            pos = wall_pos + np.array(offset, dtype=COORD_DTYPE)
            cache_manager.occupancy_cache.set_position(
                pos,
                np.array([PieceType.WALL.value, Color.WHITE], dtype=np.int8)
            )
        
        # Generate moves
        moves = generate_wall_moves(cache_manager, Color.WHITE, wall_pos)
        print(f"  Generated {len(moves)} moves")
        print(f"  ✓ No crash!")
        
        # Check if any moves would go out of bounds
        for move in moves:
            dest = move[3:]
            # Check destination anchor + offsets
            for offset in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]:
                dest_square = dest + np.array(offset, dtype=COORD_DTYPE)
                if np.any(dest_square >= SIZE) or np.any(dest_square < 0):
                    print(f"  ✗ INVALID MOVE: {move} would place block at {dest_square}")
                    return False
        
        print(f"  ✓ All moves are within bounds!")
        
    except IndexError as e:
        print(f"  ✗ FAILED with IndexError: {e}")
        return False
    except Exception as e:
        print(f"  ✗ FAILED with exception: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_wall_edge_moves()
    sys.exit(0 if success else 1)
