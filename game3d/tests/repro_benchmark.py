
import time
import numpy as np
import sys
import os

# Mock classes to avoid dependencies
class MockOccupancyCache:
    def __init__(self):
        self._occ = np.zeros((9, 9, 9), dtype=np.int8)

class MockAuraCache:
    def __init__(self):
        self._buffed_squares = np.zeros((9, 9, 9), dtype=bool)

class MockCacheManager:
    def __init__(self):
        self.occupancy_cache = MockOccupancyCache()
        self.consolidated_aura_cache = MockAuraCache()

# Import the engine (assuming it's in the path or we can import it)
# We need to add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from game3d.movement.jump_engine import JumpMovementEngine
    from game3d.common.shared_types import COORD_DTYPE
except ImportError:
    # If we can't import, we might need to copy the code here or fix paths
    # But let's assume we can run this from the project root
    pass

def benchmark():
    cache_manager = MockCacheManager()
    engine = JumpMovementEngine()
    
    # Setup data
    n_pos = 100
    positions = np.random.randint(0, 9, size=(n_pos, 3)).astype(COORD_DTYPE)
    directions = np.array([
        [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
        [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0]
    ], dtype=COORD_DTYPE)
    
    color = 1
    
    # Warmup
    engine.generate_jump_moves(cache_manager, color, positions, directions)
    
    # Benchmark Normal Batch
    start = time.time()
    for _ in range(1000):
        engine.generate_jump_moves(cache_manager, color, positions, directions)
    end = time.time()
    print(f"Batch Normal: {end - start:.4f}s")
    
    # Benchmark Buffed Batch (Force buff path)
    # Mark some squares as buffed
    cache_manager.consolidated_aura_cache._buffed_squares[:] = True
    
    start = time.time()
    for _ in range(1000):
        engine.generate_jump_moves(cache_manager, color, positions, directions)
    end = time.time()
    print(f"Batch Buffed: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
