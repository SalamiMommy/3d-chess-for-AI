

import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
from game3d.common.coord_utils import coords_to_keys
import numpy as np

def verify_overflow():
    coords = np.array([[0, 0, 1]], dtype=np.int16)
    print(f"Original: {coords}, dtype: {coords.dtype}")
    
    keys = coords_to_keys(coords)
    print(f"Result: {keys}, dtype: {keys.dtype}")
    
    correct_val = 1 << 18
    print(f"Expected: {correct_val}")
    
    if keys[0] != correct_val:
        print("OVERFLOW DETECTED!")
    else:
        print("No overflow.")

if __name__ == "__main__":
    verify_overflow()
