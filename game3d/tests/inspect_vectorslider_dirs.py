import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.pieces.pieces.vectorslider import VECTOR_DIRECTIONS

def inspect_directions():
    print(f"Total directions: {len(VECTOR_DIRECTIONS)}")
    
    target = np.array([1, 2, 0])
    found = False
    for d in VECTOR_DIRECTIONS:
        if np.array_equal(d, target):
            found = True
            break
            
    if found:
        print("Found [1, 2, 0] in VECTOR_DIRECTIONS.")
    else:
        print("MISSING [1, 2, 0] in VECTOR_DIRECTIONS!")
        
    # Also check [1, 1, 1]
    target = np.array([1, 1, 1])
    found = False
    for d in VECTOR_DIRECTIONS:
        if np.array_equal(d, target):
            found = True
            break
    if found:
        print("Found [1, 1, 1] in VECTOR_DIRECTIONS.")
    else:
        print("MISSING [1, 1, 1] in VECTOR_DIRECTIONS!")

if __name__ == "__main__":
    inspect_directions()
