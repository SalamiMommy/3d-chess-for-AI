
import time
import numpy as np
import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game3d.common.shared_types import Color, COORD_DTYPE, SIZE
from game3d.game.factory import start_game_state
from game3d.main_game import OptimizedGame3D
from game3d.attacks.check import square_attacked_by_incremental

def profile_incremental_check():
    print("Initializing Game for Incremental Check Profile...")
    gs = start_game_state()
    game = OptimizedGame3D(board=gs.board, cache=gs.cache_manager)
    
    # Warmup
    print("Warming up...")
    game.state.cache_manager.move_cache.get_legal_moves(Color.WHITE)
    
    # Setup test parameters
    target_square = np.array([4, 4, 4], dtype=COORD_DTYPE) # Center
    attacker_color = Color.BLACK
    
    # Simulate a move: White moves a Pawn?
    # We need a valid from/to coordinate that MIGHT affect the board.
    # Let's say we simulate moving a piece from (1, 4, 4) to (2, 4, 4).
    # It doesn't need to be a real move for the function, just coords.
    from_coord = np.array([1, 4, 4], dtype=COORD_DTYPE)
    to_coord = np.array([2, 4, 4], dtype=COORD_DTYPE)
    
    # Ensure cache has something
    # We need black moves cached to have an "old attack mask" etc.
    game.state.cache_manager.move_cache.get_pseudolegal_moves(Color.BLACK)
    
    print("Running profile loop...")
    iterations = 10000
    start_time = time.time()
    
    cache = game.state.cache_manager
    board = game.state.board
    
    for _ in range(iterations):
        # We call the optimized function
        # This function typically handles "In Check" incremental logic
        res = square_attacked_by_incremental(
            board,
            target_square,
            attacker_color,
            cache,
            from_coord,
            to_coord
        )
        
    end_time = time.time()
    duration = end_time - start_time
    avg_us = (duration / iterations) * 1e6
    
    print(f"Total time: {duration:.4f}s")
    print(f"Iterations: {iterations}")
    print(f"Average time per call: {avg_us:.2f} µs")
    
    # Validation of result (optional, just to ensure it ran)
    print(f"Last result: {res}")

if __name__ == "__main__":
    profile_incremental_check()
