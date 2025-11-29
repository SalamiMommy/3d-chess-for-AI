
import time
import numpy as np
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import SIZE, Color, PieceType, COORD_DTYPE
from game3d.pieces.pieces.queen import generate_queen_moves

class MockBoard:
    def __init__(self):
        self.generation = 0
    def get_initial_setup(self):
        return (np.empty((0, 3)), np.empty(0), np.empty(0))

def benchmark_slider():
    cache_manager = OptimizedCacheManager(MockBoard())
    cache_manager.occupancy_cache.clear()
    
    # Create 1000 queens
    N = 1000
    pos_list = []
    for _ in range(N):
        pos_list.append([np.random.randint(0, SIZE), np.random.randint(0, SIZE), np.random.randint(0, SIZE)])
    
    batch_pos = np.array(pos_list, dtype=COORD_DTYPE)
    
    # Sequential
    start_time = time.time()
    count = 0
    for i in range(N):
        moves = generate_queen_moves(cache_manager, Color.WHITE, batch_pos[i])
        count += len(moves)
    end_time = time.time()
    print(f"Sequential (simulated): {end_time - start_time:.4f}s")
    
    # Batch
    start_time = time.time()
    moves = generate_queen_moves(cache_manager, Color.WHITE, batch_pos)
    end_time = time.time()
    print(f"Batch: {end_time - start_time:.4f}s")
    print(f"Moves count: {len(moves)} vs {count}")

if __name__ == "__main__":
    benchmark_slider()
