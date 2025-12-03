
import time
import numpy as np
from unittest.mock import MagicMock
from game3d.movement.jump_engine import JumpMovementEngine
from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch
from game3d.common.shared_types import COORD_DTYPE, SIZE, PieceType, N_PIECE_TYPES
from game3d.game.gamestate import GameState

def run_benchmark():
    print("Benchmarking Bottlenecks: Jump Engine & Pseudolegal Batch")
    print("-" * 60)

    # --- Setup ---
    n_pieces = 1000
    positions = np.random.randint(0, SIZE, size=(n_pieces, 3)).astype(COORD_DTYPE)
    
    # Mock Cache Manager & GameState
    cache_manager = MagicMock()
    cache_manager.occupancy_cache._occ = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8)
    cache_manager.occupancy_cache._ptype = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8)
    cache_manager.occupancy_cache.get_flattened_occupancy.return_value = np.zeros(SIZE**3, dtype=np.int8)
    # Mock consolidated_aura_cache._buffed_squares
    cache_manager.consolidated_aura_cache._buffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=np.bool_)
    
    # Mock batch_get_attributes_unsafe for pseudolegal
    # Return random types and colors
    colors = np.ones(n_pieces, dtype=np.int8) # All white
    # Restrict to Knight (2) and Rook (4) to avoid complex piece logic in benchmark
    types = np.random.choice([2, 4], size=n_pieces).astype(np.int8)
    cache_manager.occupancy_cache.batch_get_attributes_unsafe.return_value = (colors, types)
    cache_manager.occupancy_cache.batch_get_attributes.return_value = (colors, types)

    game_state = MagicMock(spec=GameState)
    game_state.cache_manager = cache_manager
    game_state.color = 1

    # Jump Engine Setup
    jump_engine = JumpMovementEngine()
    directions = np.array([
        [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
        [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0]
    ], dtype=COORD_DTYPE) # Knight-like

    # Warmup
    print("Warming up...")
    jump_engine.generate_jump_moves(cache_manager, 1, positions[:10], directions)
    generate_pseudolegal_moves_batch(game_state, positions)

    # --- Benchmark Jump Engine ---
    print("\nBenchmarking generate_jump_moves (100 iterations, 1000 pieces)...")
    start = time.perf_counter()
    for _ in range(100):
        jump_engine.generate_jump_moves(cache_manager, 1, positions, directions)
    end = time.perf_counter()
    jump_time = (end - start) * 1000 / 100
    print(f"Jump Engine Time: {jump_time:.4f} ms per call")

    # --- Benchmark Pseudolegal Batch ---
    print("\nBenchmarking generate_pseudolegal_moves_batch (100 iterations, 1000 pieces)...")
    start = time.perf_counter()
    for _ in range(100):
        generate_pseudolegal_moves_batch(game_state, positions)
    end = time.perf_counter()
    pseudo_time = (end - start) * 1000 / 100
    print(f"Pseudolegal Batch Time: {pseudo_time:.4f} ms per call")

    # --- Benchmark Small Batch (Overhead) ---
    print("\nBenchmarking generate_jump_moves (1000 iterations, batch size 1)...")
    pos_single = positions[:1]
    start = time.perf_counter()
    for _ in range(1000):
        jump_engine.generate_jump_moves(cache_manager, 1, pos_single, directions)
    end = time.perf_counter()
    small_batch_time = (end - start) * 1000 / 1000
    print(f"Jump Engine Small Batch Time: {small_batch_time:.4f} ms per call")

if __name__ == "__main__":
    run_benchmark()
