
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import COORD_DTYPE, PieceType, Color, SIZE
from game3d.movement.movementmodifiers import get_range_modifier
from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.pieces.pieces.knight import KNIGHT_MOVEMENT_VECTORS
from game3d.game.gamestate import GameState
from game3d.cache.manager import OptimizedCacheManager

class MockAuraCache:
    def __init__(self):
        self._buffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
        self._debuffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=bool)

    def batch_is_buffed(self, positions, color):
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        return self._buffed_squares[x, y, z]

    def batch_is_debuffed(self, positions, color):
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        return self._debuffed_squares[x, y, z]

class MockCacheManager:
    def __init__(self):
        self.occupancy_cache = OccupancyCache()
        self.consolidated_aura_cache = MockAuraCache()
        self._effect_cache_instances = []

class MockGameState:
    def __init__(self):
        self.cache_manager = MockCacheManager()
        self.color = Color.WHITE

def benchmark():
    print("Initializing Bottleneck Benchmark...")
    state = MockGameState()
    occ = state.cache_manager.occupancy_cache
    
    # Setup board
    coords = []
    types = []
    colors = []
    
    # Create a dense board state
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                if (x + y + z) % 2 == 0:
                    coords.append([x, y, z])
                    # Mix of pieces
                    if z == 0:
                        types.append(PieceType.ARCHER.value)
                    elif z == 1:
                        types.append(PieceType.KNIGHT.value)
                    elif z == 2:
                        types.append(PieceType.PAWN.value)
                    else:
                        types.append(PieceType.WALL.value) # Filler
                    colors.append(Color.WHITE if (x+y)%2==0 else Color.BLACK)
    
    coords = np.array(coords, dtype=COORD_DTYPE)
    types = np.array(types, dtype=np.int8)
    colors = np.array(colors, dtype=np.int8)
    
    occ.rebuild(coords, types, colors)
    
    print(f"Board populated with {len(coords)} pieces.")
    
    iterations = 50
    
    # --- Benchmark 1: generate_jump_moves (Knight) ---
    print("\nBenchmarking generate_jump_moves (Knight)...")
    knight_indices = np.where(types == PieceType.KNIGHT.value)[0]
    knight_pos = coords[knight_indices]
    print(f"Processing {len(knight_pos)} knights per iteration.")
    
    jump_engine = get_jump_movement_generator()
    
    # Warmup
    if len(knight_pos) > 0:
        jump_engine.generate_jump_moves(
            state.cache_manager, Color.WHITE, knight_pos[:1], 
            KNIGHT_MOVEMENT_VECTORS, piece_type=PieceType.KNIGHT
        )
        
    start = time.time()
    for _ in range(iterations):
        jump_engine.generate_jump_moves(
            state.cache_manager, Color.WHITE, knight_pos, 
            KNIGHT_MOVEMENT_VECTORS, piece_type=PieceType.KNIGHT
        )
    end = time.time()
    print(f"generate_jump_moves (Knight): {end - start:.4f}s")
    
    # --- Benchmark 2: generate_pseudolegal_moves_batch ---
    print("\nBenchmarking generate_pseudolegal_moves_batch...")
    # Use a subset of coords for realistic batch size
    batch_size = 100
    batch_coords = coords[:batch_size]
    print(f"Processing batch of {batch_size} pieces per iteration.")
    
    # Warmup
    generate_pseudolegal_moves_batch(state, batch_coords[:1])
    
    start = time.time()
    for _ in range(iterations):
        generate_pseudolegal_moves_batch(state, batch_coords)
    end = time.time()
    print(f"generate_pseudolegal_moves_batch: {end - start:.4f}s")
    
    # --- Benchmark 3: get_range_modifier ---
    print("\nBenchmarking get_range_modifier...")
    print(f"Processing {len(coords)} positions per iteration.")
    
    start = time.time()
    for _ in range(iterations * 10): 
        get_range_modifier(state, coords)
    end = time.time()
    print(f"get_range_modifier: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
