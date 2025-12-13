
import time
import numpy as np
from numba import njit
import gc

# Mock constants
SIZE = 9
COLOR_DTYPE = np.int8
PIECE_TYPE_DTYPE = np.int8
COORD_DTYPE = np.int16

@njit(cache=True, fastmath=True)
def _extract_sparse_kernel(occ_grid, ptype_grid, out_coords, out_types, out_colors):
    count = 0
    S = occ_grid.shape[0]
    limit = out_coords.shape[0]
    for x in range(S):
        for y in range(S):
            for z in range(S):
                color = occ_grid[x, y, z]
                if color != 0:
                    if count < limit:
                        out_coords[count, 0] = x
                        out_coords[count, 1] = y
                        out_coords[count, 2] = z
                        out_types[count] = ptype_grid[x, y, z]
                        out_colors[count] = color
                    count += 1
    return count

def benchmark_export_methods():
    # Setup
    occ = np.zeros((SIZE, SIZE, SIZE), dtype=COLOR_DTYPE)
    ptype = np.zeros((SIZE, SIZE, SIZE), dtype=PIECE_TYPE_DTYPE)
    
    # Fill with 32 pieces (realistic)
    coords = []
    for _ in range(32):
        while True:
            x, y, z = np.random.randint(0, SIZE, 3)
            if occ[x, y, z] == 0:
                occ[x, y, z] = np.random.randint(1, 3)
                ptype[x, y, z] = np.random.randint(1, 7)
                coords.append(x | (y << 9) | (z << 18))
                break
                
    white_set = set(c for c in coords if occ[c & 0x1FF, (c >> 9) & 0x1FF, (c >> 18) & 0x1FF] == 1)
    black_set = set(c for c in coords if occ[c & 0x1FF, (c >> 9) & 0x1FF, (c >> 18) & 0x1FF] == 2)
    
    out_coords = np.zeros((512, 3), dtype=COORD_DTYPE)
    out_types = np.zeros(512, dtype=PIECE_TYPE_DTYPE)
    out_colors = np.zeros(512, dtype=COLOR_DTYPE)
    
    iterations = 10000
    
    # 1. Warmup and Benchmark Numba Kernel
    _extract_sparse_kernel(occ, ptype, out_coords, out_types, out_colors)
    start = time.perf_counter()
    for _ in range(iterations):
        _extract_sparse_kernel(occ, ptype, out_coords, out_types, out_colors)
    end = time.perf_counter()
    print(f"Numba Kernel: {(end - start) * 1000:.2f} ms")
    
    # 2. Benchmark Argwhere
    start = time.perf_counter()
    for _ in range(iterations):
        mask = occ != 0
        found_coords = np.argwhere(mask)
        # Not full reconstruction, just finding coords
        # To be fair, access attributes too
        x, y, z = found_coords[:, 0], found_coords[:, 1], found_coords[:, 2]
        _ = occ[x, y, z]
        _ = ptype[x, y, z]
    end = time.perf_counter()
    print(f"Numpy Argwhere: {(end - start) * 1000:.2f} ms")
    
    # 3. Benchmark Incremental Sets
    start = time.perf_counter()
    for _ in range(iterations):
        # Merge sets
        indices = list(white_set) + list(black_set)
        indices_arr = np.array(indices, dtype=np.int64) # This allocation is costly?
        indices_arr.sort()
        
        # Unpack (mock implementation of fast unpack)
        x = indices_arr & 0x1FF
        y = (indices_arr >> 9) & 0x1FF
        z = (indices_arr >> 18) & 0x1FF
        
        # Attribute lookup (vectorized)
        # Note: In reality we might use direct indexing if flat view available
        _ = occ[x, y, z]
        _ = ptype[x, y, z]
    end = time.perf_counter()
    print(f"Incremental Sets: {(end - start) * 1000:.2f} ms")

if __name__ == "__main__":
    benchmark_export_methods()
