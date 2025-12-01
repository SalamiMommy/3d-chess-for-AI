
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.cache.caches.movecache import MoveCache
from game3d.common.shared_types import SIZE, MOVE_DTYPE, COORD_DTYPE

class MockCacheManager:
    def __init__(self):
        self.board = MockBoard()
        self.occupancy_cache = MockOccupancyCache()

class MockBoard:
    def __init__(self):
        self.generation = 0

class MockOccupancyCache:
    def get_positions(self, color):
        return np.empty((0, 3), dtype=COORD_DTYPE)
    def get_priest_count(self, color):
        return 0

def benchmark_movecache():
    print("Initializing MoveCache benchmark...")
    cache_manager = MockCacheManager()
    move_cache = MoveCache(cache_manager)
    
    # Create a large set of moves
    n_moves = 10000
    moves = np.zeros((n_moves, 6), dtype=COORD_DTYPE)
    
    # Random moves within bounds
    moves[:, 3] = np.random.randint(0, SIZE, n_moves)
    moves[:, 4] = np.random.randint(0, SIZE, n_moves)
    moves[:, 5] = np.random.randint(0, SIZE, n_moves)
    
    mask = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    
    iterations = 1000
    
    print(f"Benchmarking add_moves_to_mask with {n_moves} moves over {iterations} iterations...")
    start = time.time()
    for _ in range(iterations):
        move_cache.add_moves_to_mask(mask, moves)
    end = time.time()
    print(f"add_moves_to_mask: {end - start:.4f}s")
    
    print(f"Benchmarking remove_moves_from_mask with {n_moves} moves over {iterations} iterations...")
    start = time.time()
    for _ in range(iterations):
        move_cache.remove_moves_from_mask(mask, moves)
    end = time.time()
    print(f"remove_moves_from_mask: {end - start:.4f}s")
    
    # Verify correctness
    # Add moves, check if mask is set
    mask.fill(False)
    test_moves = np.array([[0,0,0, 1,1,1], [0,0,0, 2,2,2]], dtype=COORD_DTYPE)
    move_cache.add_moves_to_mask(mask, test_moves)
    assert mask[1,1,1] == True
    assert mask[2,2,2] == True
    assert mask[0,0,0] == False
    
    move_cache.remove_moves_from_mask(mask, test_moves)
    assert mask[1,1,1] == False
    assert mask[2,2,2] == False
    
    print("Correctness verification passed.")

if __name__ == "__main__":
    benchmark_movecache()
