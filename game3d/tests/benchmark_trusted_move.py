
import sys
import os
import time
import numpy as np
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.game.gamestate import GameState
from game3d.game.turnmove import make_move, make_move_trusted, legal_moves
from game3d.common.shared_types import Color

def run_benchmark():
    print("Initializing GameState...")
    state = GameState.from_startpos()
    
    print("Generating legal moves (warmup)...")
    moves = legal_moves(state)
    print(f"Initial legal moves: {len(moves)}")
    
    if len(moves) == 0:
        print("No moves available! Benchmark aborted.")
        return

    # Benchmark Standard make_move
    start_time = time.time()
    iterations = 1000
    
    # Clone state for fairness
    test_state_1 = state.clone()
    moves_to_test = np.tile(moves, (iterations // len(moves) + 1, 1))[:iterations]
    
    print(f"\nBenchmarking make_move with {iterations} iterations...")
    safe_start = time.time()
    for i in range(iterations):
        # We just ping-pong or reset for simplicity, but actually updating state is complex
        # because the game progresses.
        # Ideally we want to apply a move, undo it, or apply a distinct sequence.
        # But undo is slow.
        # Let's clone inside the loop? No, that measures clone time.
        
        # Strategy: Apply move to a fresh clone every time? Slow.
        # Strategy: Just apply trusted move to a chain of states?
        # But games end.
        
        # Better: Measure pure function time on a fresh copy each time (overhead of copy is high but consistent)
        # Actually, let's just measure "Apply Valid Move" on a distinct clone.
        # Pre-clone N states.
        pass

    # New Strategy: Pre-allocate N clones
    print(f"Pre-cloning {iterations} states (this might take a moment)...")
    clones_safe = [state.clone() for _ in range(iterations)]
    clones_trusted = [state.clone() for _ in range(iterations)]
    
    # Safe Benchmark
    t0 = time.time()
    for i in range(iterations):
        mv = moves_to_test[i]
        make_move(clones_safe[i], mv)
    t1 = time.time()
    safe_time = t1 - t0
    safe_rate = iterations / safe_time
    print(f"safe_make_move: {safe_time:.4f}s ({safe_rate:.2f} moves/sec)")
    
    # Trusted Benchmark
    t0 = time.time()
    for i in range(iterations):
        mv = moves_to_test[i]
        make_move_trusted(clones_trusted[i], mv)
    t1 = time.time()
    trusted_time = t1 - t0
    trusted_rate = iterations / trusted_time
    print(f"trusted_make_move: {trusted_time:.4f}s ({trusted_rate:.2f} moves/sec)")
    
    speedup = trusted_rate / safe_rate
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Verification
    print("\nVerifying correctness...")
    failures = 0
    for i in range(iterations):
        # Check Zobrist
        if clones_safe[i].zkey != clones_trusted[i].zkey:
            failures += 1
            if failures <= 5:
                print(f"Mismatch at iter {i}: Safe {clones_safe[i].zkey} != Trusted {clones_trusted[i].zkey}")
    
    if failures == 0:
        print("Verification PASSED: All states result in identical Zobrist keys.")
    else:
        print(f"Verification FAILED: {failures} mismatches found.")

if __name__ == "__main__":
    run_benchmark()
