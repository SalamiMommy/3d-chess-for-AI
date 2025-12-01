
import time
import numpy as np
from numba import njit
from game3d.common.shared_types import COORD_DTYPE

# Original Numpy implementation
def calc_numpy(directions):
    current_directions = directions.copy()
    abs_dirs = np.abs(current_directions)
    max_vals = np.max(abs_dirs, axis=1, keepdims=True)
    mask = (abs_dirs == max_vals)
    sign = np.sign(current_directions)
    buffed_directions = current_directions + (sign * mask).astype(COORD_DTYPE)
    return buffed_directions

# Numba implementation (copied from my change)
@njit(cache=True, fastmath=True)
def calc_numba(directions):
    n = directions.shape[0]
    buffed = np.empty_like(directions)
    
    for i in range(n):
        dx, dy, dz = directions[i]
        
        # Calculate abs
        ax = abs(dx)
        ay = abs(dy)
        az = abs(dz)
        
        # Find max
        max_val = ax
        if ay > max_val:
            max_val = ay
        if az > max_val:
            max_val = az
            
        # Add sign to max components
        bdx = dx
        bdy = dy
        bdz = dz
        
        if ax == max_val and ax > 0:
            if dx > 0: bdx += 1
            else: bdx -= 1
            
        if ay == max_val and ay > 0:
            if dy > 0: bdy += 1
            else: bdy -= 1
            
        if az == max_val and az > 0:
            if dz > 0: bdz += 1
            else: bdz -= 1
            
        buffed[i, 0] = bdx
        buffed[i, 1] = bdy
        buffed[i, 2] = bdz
        
    return buffed

def benchmark():
    directions = np.array([
        [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
        [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0]
    ], dtype=COORD_DTYPE)
    
    # Warmup
    calc_numpy(directions)
    calc_numba(directions)
    
    iterations = 100000
    
    start = time.time()
    for _ in range(iterations):
        calc_numpy(directions)
    end = time.time()
    print(f"Numpy: {end - start:.4f}s")
    
    start = time.time()
    for _ in range(iterations):
        calc_numba(directions)
    end = time.time()
    print(f"Numba: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
