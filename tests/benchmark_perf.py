
import time
import numpy as np
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.movement.jump_engine import JumpMovementEngine
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE
from game3d.cache.manager import OptimizedCacheManager

def benchmark_occupancy_get():
    print("Benchmarking OccupancyCache.get...")
    cache = OccupancyCache()
    # Fill some spots
    for x in range(5):
        for y in range(5):
            for z in range(5):
                cache.set_position(np.array([x, y, z]), np.array([PieceType.PAWN.value, Color.WHITE.value]))
    
    start_time = time.time()
    iterations = 100000
    coord = np.array([1, 1, 1], dtype=COORD_DTYPE)
    
    for _ in range(iterations):
        cache.get(coord)
        
    end_time = time.time()
    print(f"OccupancyCache.get: {iterations} iterations in {end_time - start_time:.4f}s")
    print(f"Time per call: {(end_time - start_time) / iterations * 1e6:.2f} us")

def benchmark_jump_moves():
    print("\nBenchmarking JumpMovementEngine.generate_jump_moves...")
    
    # Mock Cache Manager
    class MockCacheManager:
        def __init__(self):
            self.occupancy_cache = OccupancyCache()
            self._effect_cache_instances = [] # No aura caches for now to isolate jump logic
            
            # Fill occupancy
            for x in range(2):
                for y in range(2):
                    self.occupancy_cache.set_position(np.array([x, y, 0]), np.array([PieceType.PAWN.value, Color.BLACK.value]))

        @property
        def consolidated_aura_cache(self):
            return None # Or a mock object if needed, but None triggers the fast path check (and skips buff logic)

    cache_manager = MockCacheManager()
    engine = JumpMovementEngine()
    
    pos = np.array([3, 3, 3], dtype=COORD_DTYPE)
    directions = np.array([
        [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
        [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0]
    ], dtype=COORD_DTYPE)
    
    start_time = time.time()
    iterations = 10000
    
    for _ in range(iterations):
        engine.generate_jump_moves(
            cache_manager,
            Color.WHITE,
            pos,
            directions,
            allow_capture=True,
            piece_type=PieceType.KNIGHT
        )
        
    end_time = time.time()
    print(f"generate_jump_moves: {iterations} iterations in {end_time - start_time:.4f}s")
    print(f"Time per call: {(end_time - start_time) / iterations * 1e6:.2f} us")

if __name__ == "__main__":
    benchmark_occupancy_get()
    benchmark_jump_moves()
