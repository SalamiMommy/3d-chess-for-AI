
import numpy as np
import sys
import os
from unittest.mock import patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.common.shared_types import PieceType, Color, SIZE, COORD_DTYPE
from game3d.pieces.pieces.wall import generate_wall_moves, _generate_wall_moves_batch_kernel

# Mock CacheManager and OccupancyCache
class MockOccupancyCache:
    def __init__(self):
        self._occ = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8)
    
    def batch_get_attributes(self, coords):
        # Return empty for all
        n = coords.shape[0]
        return np.zeros(n, dtype=int), np.zeros(n, dtype=int)

class MockAuraCache:
    def __init__(self):
        self._buffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=bool)

class MockCacheManager:
    def __init__(self):
        self.occupancy_cache = MockOccupancyCache()
        self.consolidated_aura_cache = MockAuraCache()

def test_defensive_check():
    print(f"SIZE: {SIZE}")
    cache_manager = MockCacheManager()
    wall_anchor = np.array([1, 6, 1], dtype=COORD_DTYPE)
    
    # Mark as buffed
    cache_manager.consolidated_aura_cache._buffed_squares[1, 6, 1] = True
    
    # We want to force the kernel to return an invalid move to test the defensive check.
    # We can wrap the kernel function.
    
    original_kernel = _generate_wall_moves_batch_kernel
    
    def mocked_kernel(*args, **kwargs):
        moves = original_kernel(*args, **kwargs)
        # Inject invalid move: [1, 6, 1] -> [2, 8, 0] (Dest Y=8 is invalid for 2x2 wall)
        invalid_move = np.array([[1, 6, 1, 2, 8, 0]], dtype=COORD_DTYPE)
        print(f"Injecting invalid move: {invalid_move}")
        return np.vstack([moves, invalid_move])

    # Patch the kernel in the module
    with patch('game3d.pieces.pieces.wall._generate_wall_moves_batch_kernel', side_effect=mocked_kernel):
        print("Running generate_wall_moves with injected invalid move...")
        moves = generate_wall_moves(cache_manager, Color.WHITE, wall_anchor)
        
        print(f"Generated {len(moves)} moves after filtering")
        
        # Check if invalid move is present
        invalid_present = False
        for move in moves:
            if np.array_equal(move[3:], [2, 8, 0]):
                invalid_present = True
                break
        
        if invalid_present:
            print("❌ FAILURE: Invalid move [2, 8, 0] was NOT filtered!")
            sys.exit(1)
        else:
            print("✅ SUCCESS: Invalid move [2, 8, 0] was correctly filtered.")

if __name__ == "__main__":
    test_defensive_check()
