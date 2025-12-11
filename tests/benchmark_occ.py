
import time
import numpy as np
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import SIZE, Color

def benchmark():
    print(f"Benchmarking OccupancyCache with SIZE={SIZE}")
    cache = OccupancyCache()
    
    # Fill board with some pieces
    # Simulate a game state with ~30 pieces
    np.random.seed(42)
    for _ in range(30):
        x, y, z = np.random.randint(0, SIZE, 3)
        cache.set_position_fast(np.array([x, y, z]), 1, Color.WHITE)
        
    for _ in range(30):
        x, y, z = np.random.randint(0, SIZE, 3)
        cache.set_position_fast(np.array([x, y, z]), 1, Color.BLACK)
        
    start_time = time.time()
    n_iters = 50000
    
    # Simulation of get_positions being called often with dirty cache
    for i in range(n_iters):
        # Dirty the cache
        cache._positions_dirty[0] = True
        # Get positions (triggers rebuild)
        _ = cache.get_positions(Color.WHITE)
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.4f}s")
    print(f"Time per call: {total_time/n_iters*1000:.4f}ms")
    print(f"Calls per second: {n_iters/total_time:.2f}")

if __name__ == "__main__":
    benchmark()
