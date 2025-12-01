
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import COORD_DTYPE, PieceType, Color, SIZE
from game3d.pieces.pieces.pawn import generate_pawn_moves
from game3d.movement.jump_engine import JumpMovementEngine
from game3d.cache.manager import OptimizedCacheManager
from game3d.pieces.pieces.friendlytp import generate_friendlytp_moves
from game3d.pieces.pieces.wall import generate_wall_moves

class MockCacheManager:
    def __init__(self):
        self.occupancy_cache = OccupancyCache()
        self.consolidated_aura_cache = MockAuraCache()

class MockAuraCache:
    def __init__(self):
        self._buffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=bool)

def benchmark():
    print("Initializing...")
    cache_manager = MockCacheManager()
    occ = cache_manager.occupancy_cache
    
    # Setup some pieces
    coords = []
    types = []
    colors = []
    
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                if (x + y + z) % 3 == 0:
                    coords.append([x, y, z])
                    types.append(PieceType.PAWN.value)
                    colors.append(Color.WHITE if (x+y)%2==0 else Color.BLACK)
    
    coords = np.array(coords, dtype=COORD_DTYPE)
    types = np.array(types, dtype=np.int8)
    colors = np.array(colors, dtype=np.int8)
    
    occ.rebuild(coords, types, colors)
    
    print(f"Board populated with {len(coords)} pieces.")
    
    iterations = 1000
    
    # Warmup
    print("Warming up JIT...")
    occ.batch_is_occupied_unsafe(coords[:3])
    occ.batch_is_occupied_unsafe(coords[:50])
    occ.batch_is_occupied_unsafe(coords[:200])
    occ.batch_get_attributes_unsafe(coords[:3])
    occ.batch_get_attributes(coords[:24]) # Warmup for wall moves
    
    # Warmup generate_wall_moves
    wall_pos_warmup = np.array([2, 2, 2], dtype=COORD_DTYPE)
    try:
        generate_wall_moves(cache_manager, Color.WHITE, wall_pos_warmup)
    except:
        pass
    
    # ... (rest of the code)
    # ... (rest of the code)
    
    # Benchmark batch_is_occupied_unsafe (Adaptive)
    print("\nBenchmarking batch_is_occupied_unsafe (Adaptive)...")
    
    # Tiny batch
    tiny_coords = coords[:3]
    start = time.time()
    for _ in range(iterations):
        occ.batch_is_occupied_unsafe(tiny_coords)
    end = time.time()
    print(f"batch_is_occupied_unsafe (Tiny n=3): {end - start:.4f}s")
    
    # Medium batch
    medium_coords = coords[:50]
    start = time.time()
    for _ in range(iterations):
        occ.batch_is_occupied_unsafe(medium_coords)
    end = time.time()
    print(f"batch_is_occupied_unsafe (Medium n=50): {end - start:.4f}s")
    
    # Large batch
    large_coords = coords[:200]
    start = time.time()
    for _ in range(iterations):
        occ.batch_is_occupied_unsafe(large_coords)
    end = time.time()
    print(f"batch_is_occupied_unsafe (Large n=200): {end - start:.4f}s")

    # Benchmark batch_get_attributes_unsafe (Adaptive)
    print("\nBenchmarking batch_get_attributes_unsafe (Adaptive)...")
    
    # Tiny batch
    start = time.time()
    for _ in range(iterations):
        occ.batch_get_attributes_unsafe(tiny_coords)
    end = time.time()
    print(f"batch_get_attributes_unsafe (Tiny n=3): {end - start:.4f}s")

    # Benchmark generate_jump_moves
    print("\nBenchmarking generate_jump_moves...")
    jump_engine = JumpMovementEngine()
    pawn_pos = np.array([3, 3, 3], dtype=COORD_DTYPE)
    knight_dirs = np.array([
        [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
        [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0]
    ], dtype=COORD_DTYPE)
    
    start = time.time()
    for _ in range(iterations):
        jump_engine.generate_jump_moves(
            cache_manager, Color.WHITE, pawn_pos, knight_dirs, 
            piece_type=None # Force calculation to test occupancy check
        )
    end = time.time()
    print(f"generate_jump_moves (calculated): {end - start:.4f}s")
    
    # Benchmark FriendlyTP (Stress Test)
    print("\nBenchmarking FriendlyTP (Stress Test)...")
    # FriendlyTP relies on jump moves and network building
    ftp_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    start = time.time()
    for _ in range(200): # Reduced iterations
        generate_friendlytp_moves(cache_manager, Color.WHITE, ftp_pos)
    end = time.time()
    print(f"generate_friendlytp_moves: {end - start:.4f}s")

    # Benchmark Wall Moves
    print("\nBenchmarking Wall Moves...")
    # Place a wall
    wall_pos = np.array([2, 2, 2], dtype=COORD_DTYPE)
    # Ensure it's an anchor
    occ.set_position_fast(wall_pos, PieceType.WALL.value, Color.WHITE)
    occ.set_position_fast(wall_pos + [1,0,0], PieceType.WALL.value, Color.WHITE)
    occ.set_position_fast(wall_pos + [0,1,0], PieceType.WALL.value, Color.WHITE)
    occ.set_position_fast(wall_pos + [1,1,0], PieceType.WALL.value, Color.WHITE)
    
    start = time.time()
    for _ in range(iterations):
        generate_wall_moves(cache_manager, Color.WHITE, wall_pos)
    end = time.time()
    print(f"generate_wall_moves: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
