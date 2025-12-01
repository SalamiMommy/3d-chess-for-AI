
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
from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch
from game3d.game.gamestate import GameState
from game3d.cache.manager import OptimizedCacheManager

class MockAuraCache:
    def __init__(self):
        self._buffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=bool)

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
    print("Initializing Benchmark V2...")
    state = MockGameState()
    occ = state.cache_manager.occupancy_cache
    
    # Setup board with mixed pieces
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
                        types.append(PieceType.PAWN.value)
                    elif z == 1:
                        types.append(PieceType.KNIGHT.value)
                    else:
                        types.append(PieceType.ROOK.value)
                    colors.append(Color.WHITE if (x+y)%2==0 else Color.BLACK)
    
    coords = np.array(coords, dtype=COORD_DTYPE)
    types = np.array(types, dtype=np.int8)
    colors = np.array(colors, dtype=np.int8)
    
    occ.rebuild(coords, types, colors)
    
    print(f"Board populated with {len(coords)} pieces.")
    
    # --- WARMUP ---
    print("Warming up JIT...")
    # Warmup Jump Engine
    jump_engine = JumpMovementEngine()
    knight_indices = np.where(types == PieceType.KNIGHT.value)[0]
    if len(knight_indices) > 0:
        k_pos = coords[knight_indices[:1]]
        k_dirs = np.array([[1, 2, 0]], dtype=COORD_DTYPE)
        jump_engine.generate_jump_moves(state.cache_manager, Color.WHITE, k_pos, k_dirs)

    # Warmup Pawn
    pawn_indices = np.where(types == PieceType.PAWN.value)[0]
    if len(pawn_indices) > 0:
        p_pos = coords[pawn_indices[:1]]
        generate_pawn_moves(state.cache_manager, Color.WHITE, p_pos)

    # Warmup Pseudolegal
    generate_pseudolegal_moves_batch(state, coords[:10])
    
    iterations = 100
    
    # --- Benchmark 1: generate_jump_moves (Batch) ---
    print("\nBenchmarking generate_jump_moves (Batch)...")
    jump_engine = JumpMovementEngine()
    
    # Create a large batch of positions (e.g., all knights)
    knight_indices = np.where(types == PieceType.KNIGHT.value)[0]
    knight_pos = coords[knight_indices]
    
    knight_dirs = np.array([
        [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
        [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0]
    ], dtype=COORD_DTYPE)
    
    print(f"Processing {len(knight_pos)} knights per iteration.")
    
    start = time.time()
    for _ in range(iterations):
        jump_engine.generate_jump_moves(
            state.cache_manager, Color.WHITE, knight_pos, knight_dirs
        )
    end = time.time()
    print(f"generate_jump_moves (Batch): {end - start:.4f}s")
    
    # --- Benchmark 2: generate_pawn_moves (Batch) ---
    print("\nBenchmarking generate_pawn_moves (Batch)...")
    pawn_indices = np.where(types == PieceType.PAWN.value)[0]
    pawn_pos = coords[pawn_indices]
    
    print(f"Processing {len(pawn_pos)} pawns per iteration.")
    
    start = time.time()
    for _ in range(iterations):
        generate_pawn_moves(state.cache_manager, Color.WHITE, pawn_pos)
    end = time.time()
    print(f"generate_pawn_moves (Batch): {end - start:.4f}s")
    
    # --- Benchmark 3: generate_pseudolegal_moves_batch (Mixed) ---
    print("\nBenchmarking generate_pseudolegal_moves_batch (Mixed)...")
    # Use all coords
    print(f"Processing {len(coords)} mixed pieces per iteration.")
    
    start = time.time()
    for _ in range(iterations):
        generate_pseudolegal_moves_batch(state, coords)
    end = time.time()
    print(f"generate_pseudolegal_moves_batch (Mixed): {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
