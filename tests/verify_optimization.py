
import sys
import os
import time
import numpy as np
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath("/home/salamimommy/Documents/code/3d-chess-for-AI"))

from game3d.game.factory import start_game_state
from game3d.game.turnmove import make_move, undo_move, legal_moves
from game3d.common.shared_types import MOVE_DTYPE, MAX_HISTORY_SIZE

def verify_optimizations():
    print("Initializing game state...")
    game_state = start_game_state()
    
    # 1. Verify History is Deque
    print(f"History type: {type(game_state.history)}")
    if not isinstance(game_state.history, deque):
        print("FAIL: History is not a deque")
        return False
    print("PASS: History is a deque")
    
    # 2. Run a few moves
    print("\nRunning moves...")
    start_time = time.perf_counter()
    
    for i in range(10):
        moves = legal_moves(game_state)
        if moves.size == 0:
            print("No legal moves!")
            break
            
        # Pick first move
        mv = moves[0]
        game_state = make_move(game_state, mv)
        
        # Verify history grew
        if len(game_state.history) != i + 1:
            print(f"FAIL: History length mismatch. Expected {i+1}, got {len(game_state.history)}")
            return False
            
    end_time = time.perf_counter()
    print(f"Made 10 moves in {end_time - start_time:.4f}s")
    
    # 3. Verify Undo
    print("\nTesting Undo...")
    original_len = len(game_state.history)
    game_state = undo_move(game_state)
    
    if len(game_state.history) != original_len - 1:
        print(f"FAIL: Undo did not reduce history length. Expected {original_len - 1}, got {len(game_state.history)}")
        return False
    print("PASS: Undo works")
    
    # 4. Verify History Array Property
    print("\nTesting history_array property...")
    hist_arr = game_state.history_array
    if not isinstance(hist_arr, np.ndarray):
        print("FAIL: history_array is not numpy array")
        return False
    if hist_arr.shape[0] != len(game_state.history):
        print("FAIL: history_array size mismatch")
        return False
    print("PASS: history_array works")
    
    print("\nAll verification checks passed!")
    return True

if __name__ == "__main__":
    try:
        if verify_optimizations():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
