
import time
import numpy as np
import matplotlib.pyplot as plt
from game3d.movement.jump_engine import JumpMovementEngine, _generate_jump_moves_batch_unified
from game3d.common.shared_types import COORD_DTYPE, SIZE
from unittest.mock import MagicMock

# --- Benchmark ---

def run_benchmark():
    print("Benchmarking Jump Moves: Adaptive vs Parallel Kernel")
    print("-" * 50)
    
    # Setup
    directions = np.array([
        [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
        [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0],
        [0, 1, 2], [0, 2, 1], [0, -1, 2], [0, -2, 1],
        [0, 1, -2], [0, 2, -1], [0, -1, -2], [0, -2, -1],
        [1, 0, 2], [2, 0, 1], [-1, 0, 2], [-2, 0, 1],
        [1, 0, -2], [2, 0, -1], [-1, 0, -2], [-2, 0, -1]
    ], dtype=COORD_DTYPE) # Knight moves (24 directions)
    
    # Mock Cache Manager
    cache_manager = MagicMock()
    cache_manager.occupancy_cache._occ = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8)
    cache_manager.consolidated_aura_cache._buffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=np.bool_)
    
    engine = JumpMovementEngine()
    
    # Warmup
    dummy_pos = np.array([[4, 4, 4]], dtype=COORD_DTYPE)
    engine.generate_jump_moves(cache_manager, 1, dummy_pos, directions)
    
    batch_sizes = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000]
    
    for n in batch_sizes:
        # Generate random positions
        positions = np.random.randint(0, SIZE, size=(n, 3)).astype(COORD_DTYPE)
        
        # Pure Parallel Kernel (what we had before)
        start = time.perf_counter()
        for _ in range(100):
            _generate_jump_moves_batch_unified(
                positions, directions, 
                cache_manager.consolidated_aura_cache._buffed_squares, 
                cache_manager.occupancy_cache._occ, 
                True, 1
            )
        end = time.perf_counter()
        parallel_time = (end - start) / 100 * 1000 # ms
        
        # Adaptive (via Engine)
        start = time.perf_counter()
        for _ in range(100):
            engine.generate_jump_moves(cache_manager, 1, positions, directions)
        end = time.perf_counter()
        adaptive_time = (end - start) / 100 * 1000 # ms
        
        ratio = parallel_time / adaptive_time
        winner = "Adaptive" if adaptive_time < parallel_time else "Parallel"
        
        print(f"Batch: {n:<5} | Parallel: {parallel_time:.4f} ms | Adaptive: {adaptive_time:.4f} ms | Speedup: {ratio:.2f}x ({winner})")

if __name__ == "__main__":
    run_benchmark()
