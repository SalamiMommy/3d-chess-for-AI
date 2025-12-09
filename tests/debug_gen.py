
import numpy as np
import logging
import sys
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, COORD_DTYPE
from game3d.movement.generator import generate_legal_moves
from game3d.core.buffer import state_to_buffer
from game3d.core.api import generate_legal_moves as generate_legal_moves_functional

# Setup logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def debug_generation():
    print("Initializing Board...")
    board = Board()
    state = GameState(board, Color.WHITE)
    
    print(f"State COlor: {state.color}")
    print(f"Occupied count: {len(state.cache_manager.occupancy_cache.get_all_occupied_vectorized()[0])}")
    
    # 2. Test Functional Generator directly
    print("\n--- Testing Functional Generator ---")
    buffer = state_to_buffer(state)
    moves_func = generate_legal_moves_functional(buffer)
    print(f"Functional Moves shape: {moves_func.shape}")
    
    # 3. Test High-Level Generator (which has the filtering logic)
    print("\n--- Testing High-Level Generator ---")
    state.cache_manager.move_cache.invalidate() # ensure fresh generation
    moves_hl = generate_legal_moves(state)
    print(f"High-Level Moves shape: {moves_hl.shape}")
    
    if moves_hl.size > 0:
        print(f"Sample move: {moves_hl[0]}")
    else:
        print("NO MOVES GENERATED (High-Level)")

if __name__ == "__main__":
    debug_generation()
