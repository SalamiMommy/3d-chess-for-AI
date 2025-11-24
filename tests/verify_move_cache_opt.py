
import time
import numpy as np
from game3d.cache.caches.movecache import create_move_cache
from game3d.common.shared_types import MOVE_DTYPE

class MockCacheManager:
    def __init__(self):
        self.board = type('obj', (object,), {'generation': 0})
        self.occupancy_cache = type('obj', (object,), {'get_positions': lambda c: []})

def verify_move_cache():
    cm = MockCacheManager()
    cache = create_move_cache(cm)
    
    print("Testing MoveCache Performance...")
    
    # Generate dummy moves
    moves = np.zeros((10, 6), dtype=MOVE_DTYPE)
    for i in range(10):
        moves[i] = (0, 0, 0, i, i, i)
        
    start_time = time.perf_counter()
    
    # Simulate heavy usage
    for i in range(10000):
        cache.store_piece_moves(0, i, moves)
        
    duration = time.perf_counter() - start_time
    print(f"Stored 10,000 piece moves in {duration:.4f}s")
    
    # Verify LRU
    stats = cache.get_statistics()
    print(f"Cache size: {stats['piece_moves_cache_size']}")
    print(f"Prune operations: {stats['prune_operations']}")
    
    assert stats['piece_moves_cache_size'] <= 1000, "Cache size exceeded limit"
    assert stats['prune_operations'] > 0, "Pruning did not trigger"
    
    # Verify retrieval
    retrieved = cache.get_piece_moves(0, 9999)
    assert len(retrieved) == 10, "Failed to retrieve moves"
    
    print("Verification Passed!")

if __name__ == "__main__":
    verify_move_cache()
