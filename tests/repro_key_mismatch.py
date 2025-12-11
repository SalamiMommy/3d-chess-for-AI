
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE, SIZE

def test_key_mismatch():
    print("Initializing OccupancyCache...")
    cache = OccupancyCache()
    
    # Create a test coordinate that is likely to fail if decoded incorrectly
    # (3, 3, 3) -> 
    # Bit packing: 3 | (3 << 9) | (3 << 18) = 3 | 1536 | 786432 = 787971
    # Base-9 decoding of 787971:
    # z = 787971 // 81 = 9728 (buffer overflow!)
    
    # Let's try (1, 1, 1) to start simple
    # Bit keys: 1 | 512 | 262144 = 262657
    # Base 9: 262657 // 81 = 3242 (overflow)
    
    coords = np.array([[1, 1, 1], [3, 4, 5]], dtype=COORD_DTYPE)
    pieces = np.array([
        [PieceType.PAWN, Color.WHITE],
        [PieceType.KNIGHT, Color.WHITE]
    ], dtype=np.int8)
    
    print(f"Setting positions: \n{coords}")
    
    # This calls coords_to_keys internally which uses bit packing
    cache.batch_set_positions(coords, pieces)
    
    print("Calling get_positions(Color.WHITE)...")
    try:
        # This currently uses base-9 decoding on the bit-packed keys
        recovered_coords = cache.get_positions(Color.WHITE)
        
        print(f"Recovered coordinates: \n{recovered_coords}")
        
        # Check for correctness
        if recovered_coords.shape != coords.shape:
             print(f"FAIL: Shape mismatch. Expected {coords.shape}, got {recovered_coords.shape}")
             return
             
        # Sort both for comparison
        # (Assuming simple sorting works for these small arrays)
        coords_sorted = coords[np.lexsort(coords.T[::-1])]
        recovered_sorted = recovered_coords[np.lexsort(recovered_coords.T[::-1])]
        
        if np.array_equal(coords_sorted, recovered_sorted):
            print("SUCCESS: Coordinates match!")
        else:
            print("FAIL: Mismatch!")
            print(f"Expected:\n{coords_sorted}")
            print(f"Got:\n{recovered_sorted}")
            
            # Show what the bad values are to confirm the hypothesis
            if np.any(recovered_sorted >= SIZE):
                print("FAIL: Recovered coordinates are out of bounds (confirming bug)")
                
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_key_mismatch()
