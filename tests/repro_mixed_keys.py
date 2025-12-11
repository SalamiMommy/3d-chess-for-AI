
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE

def test_mixed_keys():
    print("Initializing OccupancyCache...")
    cache = OccupancyCache()
    
    # 1. Use batch_set_positions (uses bit-packing)
    # (0, 0, 0) -> Key 0
    coords = np.array([[0, 0, 0]], dtype=COORD_DTYPE)
    pieces = np.array([[PieceType.PAWN, Color.WHITE]], dtype=np.int8)
    cache.batch_set_positions(coords, pieces)
    
    # Verify (0,0,0) is correct
    pos = cache.get_positions(Color.WHITE)
    print(f"Post-batch positions:\n{pos}")
    if not np.array_equal(pos, coords):
        print("FAIL: batch_set_positions failed basic sanity")
        return

    # 2. Use set_position_fast (uses base-9 arithmetic currently)
    # (1, 1, 0)
    # Base-9 key: 1 + 9 = 10
    # Bit-packed interpretation of 10: (10, 0, 0) -> OUT OF BOUNDS for SIZE=9
    
    coord_single = np.array([1, 1, 0], dtype=COORD_DTYPE)
    print(f"Calling set_position_fast with {coord_single}")
    # Force add a BLACK piece
    cache.set_position_fast(coord_single, PieceType.PAWN, Color.BLACK)
    
    # 3. Retrieve positions for BLACK
    print("Retrieving BLACK positions...")
    try:
        black_pos = cache.get_positions(Color.BLACK)
        print(f"Recovered Black positions:\n{black_pos}")
        
        expected = np.array([[1, 1, 0]], dtype=COORD_DTYPE)
        
        if np.array_equal(black_pos, expected):
            print("SUCCESS: Coordinates match")
        else:
            print("FAIL: Coordinates do not match")
            # If we see [10, 0, 0], it confirms the bug
            if np.any(black_pos[:, 0] == 10):
                print("CONFIRMED BUG: Recovered [10, 0, 0] instead of [1, 1, 0]")
                
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mixed_keys()
