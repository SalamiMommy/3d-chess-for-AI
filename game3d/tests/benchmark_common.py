
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.common.coord_utils import in_bounds_vectorized
from game3d.common.validation import validate_move_basic, ensure_coords
from game3d.common.shared_types import SIZE, COORD_DTYPE, Color

# Mock game state and cache
class MockOccupancyCache:
    def get_color_at(self, x, y, z):
        return 1 # White

    def batch_get_attributes(self, coords):
        return np.ones(len(coords)), np.ones(len(coords))

class MockCacheManager:
    def __init__(self):
        self.occupancy_cache = MockOccupancyCache()

class MockGameState:
    def __init__(self):
        self.cache_manager = MockCacheManager()
        self.color = 1

class MockMove:
    def __init__(self, from_coord, to_coord):
        self.from_coord = from_coord
        self.to_coord = to_coord

def benchmark_in_bounds():
    print("Benchmarking in_bounds_vectorized...")
    
    # Tiny batch
    coords_tiny = np.array([[1, 1, 1], [2, 2, 2]], dtype=COORD_DTYPE)
    start_time = time.time()
    for _ in range(100000):
        in_bounds_vectorized(coords_tiny)
    end_time = time.time()
    print(f"Tiny batch (n=2) 100k iters: {end_time - start_time:.4f}s")

    # Single coord (1D)
    coord_single = np.array([1, 1, 1], dtype=COORD_DTYPE)
    start_time = time.time()
    for _ in range(100000):
        in_bounds_vectorized(coord_single)
    end_time = time.time()
    print(f"Single coord (1D) 100k iters: {end_time - start_time:.4f}s")

def benchmark_validate_move():
    print("\nBenchmarking validate_move_basic...")
    game_state = MockGameState()
    move = MockMove(np.array([1, 1, 1], dtype=COORD_DTYPE), np.array([2, 2, 2], dtype=COORD_DTYPE))
    
    start_time = time.time()
    for _ in range(100000):
        validate_move_basic(game_state, move)
    end_time = time.time()
    print(f"Single move validation 100k iters: {end_time - start_time:.4f}s")

def benchmark_ensure_coords():
    print("\nBenchmarking ensure_coords...")
    
    # Numpy array input (fast path)
    coords_np = np.array([[1, 1, 1]], dtype=COORD_DTYPE)
    start_time = time.time()
    for _ in range(100000):
        ensure_coords(coords_np)
    end_time = time.time()
    print(f"Numpy array input 100k iters: {end_time - start_time:.4f}s")
    
    # List input
    coords_list = [1, 1, 1]
    start_time = time.time()
    for _ in range(100000):
        ensure_coords(coords_list)
    end_time = time.time()
    print(f"List input 100k iters: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    benchmark_in_bounds()
    benchmark_validate_move()
    benchmark_ensure_coords()
