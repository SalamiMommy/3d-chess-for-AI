
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import COORD_DTYPE, PieceType, Color, SIZE
from game3d.movement.movementmodifiers import get_range_modifier
from game3d.pieces.pieces.archer import generate_archer_moves
from game3d.pieces.pieces.friendlytp import generate_friendlytp_moves
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
    print("Initializing Benchmark V3...")
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
                        types.append(PieceType.FRIENDLYTELEPORTER.value)
                    else:
                        types.append(PieceType.PAWN.value) # Filler
                    colors.append(Color.WHITE if (x+y)%2==0 else Color.BLACK)
    
    coords = np.array(coords, dtype=COORD_DTYPE)
    types = np.array(types, dtype=np.int8)
    colors = np.array(colors, dtype=np.int8)
    
    occ.rebuild(coords, types, colors)
    
    print(f"Board populated with {len(coords)} pieces.")
    
    iterations = 100
    
    # --- Benchmark 1: get_range_modifier ---
    print("\nBenchmarking get_range_modifier...")
    # Use all coords
    print(f"Processing {len(coords)} positions per iteration.")
    
    start = time.time()
    for _ in range(iterations * 10): # More iterations as it's fast
        get_range_modifier(state, coords)
    end = time.time()
    print(f"get_range_modifier: {end - start:.4f}s")
    
    # --- Benchmark 2: generate_archer_moves ---
    print("\nBenchmarking generate_archer_moves...")
    archer_indices = np.where(types == PieceType.ARCHER.value)[0]
    archer_pos = coords[archer_indices]
    print(f"Processing {len(archer_pos)} archers per iteration.")
    
    # Warmup
    if len(archer_pos) > 0:
        generate_archer_moves(state.cache_manager, Color.WHITE, archer_pos[:1])
        
    start = time.time()
    for _ in range(iterations):
        generate_archer_moves(state.cache_manager, Color.WHITE, archer_pos)
    end = time.time()
    print(f"generate_archer_moves: {end - start:.4f}s")
    
    # --- Benchmark 3: generate_friendlytp_moves ---
    print("\nBenchmarking generate_friendlytp_moves...")
    ftp_indices = np.where(types == PieceType.FRIENDLYTELEPORTER.value)[0]
    ftp_pos = coords[ftp_indices]
    print(f"Processing {len(ftp_pos)} friendly teleporters per iteration.")
    
    # Warmup
    if len(ftp_pos) > 0:
        generate_friendlytp_moves(state.cache_manager, Color.WHITE, ftp_pos[:1])
        
    start = time.time()
    for _ in range(iterations):
        generate_friendlytp_moves(state.cache_manager, Color.WHITE, ftp_pos)
    end = time.time()
    print(f"generate_friendlytp_moves: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
