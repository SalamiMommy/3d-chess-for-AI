import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.cache.caches.movecache import MoveCache, create_move_cache
from game3d.cache.manager import OptimizedCacheManager as CacheManager, get_cache_manager
from game3d.common.shared_types import Color, SIZE, BOOL_DTYPE

class MockBoard:
    def __init__(self):
        self.generation = 1
    def get_initial_setup(self):
        # Return empty arrays for coords, types, colors
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int8)
        )

def verify_fix():
    print("Verifying MoveCache.get_attack_mask fix...")
    
    # Setup
    board = MockBoard()
    try:
        cache_mgr = CacheManager(board)
    except Exception as e:
        print(f"Failed to init CacheManager: {e}")
        # Try finding minimal init
        # CacheManager usually needs a board
        return False

    # Ensure move_cache is created (CacheManager might do it)
    if not hasattr(cache_mgr, 'move_cache') or cache_mgr.move_cache is None:
        move_cache = create_move_cache(cache_mgr)
        cache_mgr.move_cache = move_cache
    else:
        move_cache = cache_mgr.move_cache
    
    # Store some attacks
    # Example: Piece at (0,0,0) attacking (1,1,1)
    attacked_squares = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
    
    print("Storing attacks...")
    move_cache.store_attack(Color.WHITE.value, attacked_squares)
    
    # Verify get_attack_mask
    print("Calling get_attack_mask...")
    try:
        mask = move_cache.get_attack_mask(Color.WHITE.value)
    except AttributeError as e:
        print(f"FAILED: AttributeError: {e}")
        return False
        
    if mask is None:
        print("FAILED: Mask is None (Should strictly be array if dirty/missing handled or stored?)")
        # Wait, if I explicitly stored it, it should be clean?
        # store_attack sets _bitboard_dirty = False
        print("Debug: Bitboard dirty?", move_cache._bitboard_dirty)
        return False
        
    if not isinstance(mask, np.ndarray):
        print(f"FAILED: Mask is not np.ndarray, got {type(mask)}")
        return False
        
    # Check values
    if not mask[1, 1, 1] or not mask[2, 2, 2]:
        print("FAILED: Mask missing set bits")
        return False
        
    if mask[0, 0, 0]:
        print("FAILED: Mask has extra set bits")
        return False
        
    print("SUCCESS: get_attack_mask works correctly!")
    return True

if __name__ == "__main__":
    success = verify_fix()
    sys.exit(0 if success else 1)
