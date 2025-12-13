
import numpy as np
from numba import njit

# Mock INDEX_DTYPE same as shared_types (will check file to confirm, initially assume int64)
INDEX_DTYPE = np.int64 

@njit
def _coords_to_keys(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    keys = np.empty(n, dtype=INDEX_DTYPE)
    for i in range(n):
        x, y, z = coords[i]
        # Potential overflow if x,y,z remain int16 during shift
        keys[i] = x | (y << 9) | (z << 18)
    return keys

def test_overflow():
    coords = np.array([[0, 0, 1]], dtype=np.int16)
    keys = _coords_to_keys(coords)
    expected = 1 << 18
    print(f"Key: {keys[0]}, Expected: {expected}")
    if keys[0] != expected:
        print("OVERFLOW DETECTED in scalar Numba loop!")
    else:
        print("No overflow in scalar Numba loop.")

if __name__ == "__main__":
    test_overflow()
