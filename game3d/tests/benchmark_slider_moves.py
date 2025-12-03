
import time
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from game3d.movement.slider_engine import _generate_all_slider_moves_batch
from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED

# --- Define Serial Kernel for Comparison ---

@njit(cache=True, fastmath=True, parallel=False)
def _generate_all_slider_moves_batch_serial(
    color: int,
    positions: np.ndarray,
    directions: np.ndarray,
    max_distances: np.ndarray,
    flattened: np.ndarray,
    ignore_occupancy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Serial version of slider move generation for BATCH of positions."""
    n_pos = positions.shape[0]
    n_dirs = directions.shape[0]
    
    # Pass 1: Count moves per piece
    counts = np.zeros(n_pos, dtype=np.int32)
    
    for i in range(n_pos):
        px, py, pz = positions[i]
        max_dist = max_distances[i]
        count = 0
        
        for d in range(n_dirs):
            dx, dy, dz = directions[d]
            
            if dx == 0 and dy == 0 and dz == 0:
                continue
                
            current_x = px + dx
            current_y = py + dy
            current_z = pz + dz
            
            for _ in range(max_dist):
                if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                    break
                
                idx = current_x + SIZE * current_y + SIZE_SQUARED * current_z
                occupant = flattened[idx]
                
                if occupant == 0:
                    count += 1
                else:
                    if ignore_occupancy:
                        count += 1
                    else:
                        if occupant != color:
                            count += 1
                        break
                
                current_x += dx
                current_y += dy
                current_z += dz
        
        counts[i] = count
        
    # Pass 2: Calculate offsets
    total_moves = np.sum(counts)
    offsets = np.zeros(n_pos, dtype=np.int32)
    current_offset = 0
    for i in range(n_pos):
        offsets[i] = current_offset
        current_offset += counts[i]
        
    # Pass 3: Fill moves
    moves = np.empty((total_moves, 6), dtype=COORD_DTYPE)
    
    for i in range(n_pos):
        write_idx = offsets[i]
        px, py, pz = positions[i]
        max_dist = max_distances[i]
        
        for d in range(n_dirs):
            dx, dy, dz = directions[d]
            
            if dx == 0 and dy == 0 and dz == 0:
                continue
                
            current_x = px + dx
            current_y = py + dy
            current_z = pz + dz
            
            for _ in range(max_dist):
                if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                    break
                
                idx = current_x + SIZE * current_y + SIZE_SQUARED * current_z
                occupant = flattened[idx]
                
                should_write = False
                if occupant == 0:
                    should_write = True
                else:
                    if ignore_occupancy:
                        should_write = True
                    else:
                        if occupant != color:
                            should_write = True
                        # Break comes after writing
                
                if should_write:
                    moves[write_idx, 0] = px
                    moves[write_idx, 1] = py
                    moves[write_idx, 2] = pz
                    moves[write_idx, 3] = current_x
                    moves[write_idx, 4] = current_y
                    moves[write_idx, 5] = current_z
                    write_idx += 1
                
                if occupant != 0:
                    if not ignore_occupancy:
                        break
                
                current_x += dx
                current_y += dy
                current_z += dz
                
    return moves, np.empty(0, dtype=np.bool_)

# --- Benchmark ---

def run_benchmark():
    print("Benchmarking Slider Moves: Serial vs Parallel")
    print("-" * 50)
    
    # Setup
    directions = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=COORD_DTYPE) # Rook moves (6 directions)
    
    flattened = np.zeros(SIZE**3, dtype=np.int8)
    
    # Warmup
    dummy_pos = np.array([[4, 4, 4]], dtype=COORD_DTYPE)
    dummy_dist = np.array([8], dtype=np.int32)
    _generate_all_slider_moves_batch(1, dummy_pos, directions, dummy_dist, flattened, False)
    _generate_all_slider_moves_batch_serial(1, dummy_pos, directions, dummy_dist, flattened, False)
    
    batch_sizes = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000]
    
    for n in batch_sizes:
        # Generate random positions
        positions = np.random.randint(0, SIZE, size=(n, 3)).astype(COORD_DTYPE)
        max_dists = np.full(n, 8, dtype=np.int32)
        
        # Parallel
        start = time.perf_counter()
        for _ in range(100):
            _generate_all_slider_moves_batch(1, positions, directions, max_dists, flattened, False)
        end = time.perf_counter()
        parallel_time = (end - start) / 100 * 1000 # ms
        
        # Serial
        start = time.perf_counter()
        for _ in range(100):
            _generate_all_slider_moves_batch_serial(1, positions, directions, max_dists, flattened, False)
        end = time.perf_counter()
        serial_time = (end - start) / 100 * 1000 # ms
        
        ratio = serial_time / parallel_time
        winner = "Parallel" if parallel_time < serial_time else "Serial"
        
        print(f"Batch: {n:<5} | Serial: {serial_time:.4f} ms | Parallel: {parallel_time:.4f} ms | Speedup: {ratio:.2f}x ({winner})")

if __name__ == "__main__":
    run_benchmark()
