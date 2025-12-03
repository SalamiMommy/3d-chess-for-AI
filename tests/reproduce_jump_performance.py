import time
import numpy as np
import os
from game3d.movement.jump_engine import JumpMovementEngine, _load_precomputed_moves, _PRECOMPUTED_MOVES
from game3d.common.shared_types import PieceType, SIZE, SIZE_SQUARED
from game3d.cache.manager import OptimizedCacheManager
from game3d.game.gamestate import GameState

def benchmark_jump_moves():
    print("Initializing benchmark...")
    
    # Ensure precomputed moves are loaded
    _load_precomputed_moves()
    print(f"Precomputed moves loaded: {list(_PRECOMPUTED_MOVES.keys())}")
    
    # Setup
    game = GameState.from_startpos()
    cache_manager = OptimizedCacheManager(game.board)
    engine = JumpMovementEngine()
    
    # Create a batch of positions (e.g., 1000 Knights)
    n_pieces = 1000
    positions = np.random.randint(0, SIZE, size=(n_pieces, 3))
    
    # Knight directions
    directions = np.array([
        [1, 2, 0], [1, -2, 0], [-1, 2, 0], [-1, -2, 0],
        [2, 1, 0], [2, -1, 0], [-2, 1, 0], [-2, -1, 0],
        [0, 1, 2], [0, 1, -2], [0, -1, 2], [0, -1, -2],
        [0, 2, 1], [0, 2, -1], [0, -2, 1], [0, -2, -1],
        [1, 0, 2], [1, 0, -2], [-1, 0, 2], [-1, 0, -2],
        [2, 0, 1], [2, 0, -1], [-2, 0, 1], [-2, 0, -1]
    ], dtype=np.int8)
    
    # Run benchmark
    print(f"Benchmarking generate_jump_moves with {n_pieces} pieces...")
    
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        # We pass piece_type=PieceType.KNIGHT to simulate what should happen
        # Currently, the batch path ignores this
        moves = engine.generate_jump_moves(
            cache_manager, 
            1, # Color
            positions, 
            directions, 
            allow_capture=True, 
            piece_type=PieceType.KNIGHT
        )
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per call: {avg_time:.6f}s")
    print(f"Moves generated: {len(moves)}")

if __name__ == "__main__":
    benchmark_jump_moves()
