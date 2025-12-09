
import time
import numpy as np
import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game3d.main_game import OptimizedGame3D as Game
from game3d.cache.manager import get_cache_manager
from game3d.common.shared_types import Color, COORD_DTYPE

from game3d.game.factory import start_game_state
from game3d.main_game import OptimizedGame3D

def benchmark_affected_pieces():
    print("Initializing Game...")
    # Create valid game state using factory
    gs = start_game_state()
    # Create game wrapper
    game = OptimizedGame3D(board=gs.board, cache=gs.cache_manager)
    
    # Force cache initialization
    print("Warming up cache...")
    for _ in range(3):
        # Use game.state.cache_manager directly as Game methods wrap it
        game.submit_move
        # Just accessing properties warms up nothing, need legal moves
        game.state.cache_manager.move_cache.get_legal_moves(Color.WHITE)
    
    print("Running benchmark for 'get_pieces_affected_by_move'...")
    
    cache = game.cache_manager
    move_cache = cache.move_cache
    
    # Create some dummy moves to test
    # Moving pieces helps populate the cache if they weren't already
    
    # We will simulate many calls to get_pieces_affected_by_move
    # using valid coordinates
    
    from_coord = np.array([0, 0, 0], dtype=COORD_DTYPE) # Assuming some pieces are here
    to_coord = np.array([3, 3, 3], dtype=COORD_DTYPE)
    
    iterations = 50000
    start_time = time.time()
    
    for _ in range(iterations):
        # Alternate colors to exercise both maps if implemented
        color = Color.WHITE if _ % 2 == 0 else Color.BLACK
        move_cache.get_pieces_affected_by_move(from_coord, to_coord, color)
        
    end_time = time.time()
    duration = end_time - start_time
    avg_time_us = (duration / iterations) * 1e6
    
    print(f"Total time: {duration:.4f}s")
    print(f"Iterations: {iterations}")
    print(f"Average time per call: {avg_time_us:.2f} µs")
    
if __name__ == "__main__":
    benchmark_affected_pieces()
