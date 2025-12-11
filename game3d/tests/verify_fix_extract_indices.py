
import numpy as np
from game3d.cache.caches.movecache import _extract_bits_indices
import sys

def test_extract_bits_indices():
    print("Testing _extract_bits_indices...")
    blocks = np.zeros(12, dtype=np.uint64)
    # Set some bits
    blocks[0] = np.uint64(1) # Bit 0
    blocks[1] = np.uint64(2) # Bit 65 (64*1 + 1)
    
    indices = _extract_bits_indices(blocks)
    
    if indices is None:
        print("FAIL: _extract_bits_indices returned None")
        sys.exit(1)
        
    print(f"Indices returned: {indices}")
    
    if len(indices) != 2:
        print(f"FAIL: Expected 2 indices, got {len(indices)}")
        sys.exit(1)
        
    if 0 not in indices or 65 not in indices:
        print("FAIL: Incorrect indices returned")
        sys.exit(1)
        
    print("SUCCESS: _extract_bits_indices works correctly.")

if __name__ == "__main__":
    test_extract_bits_indices()
