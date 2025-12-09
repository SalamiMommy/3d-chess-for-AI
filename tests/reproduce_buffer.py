
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.getcwd()))

from game3d.game.gamestate import GameState
from game3d.core.buffer import state_to_buffer

def benchmark():
    print("Initializing GameState...")
    state = GameState.from_startpos()
    
    # Warmup
    print("Warming up...")
    for _ in range(100):
        _ = state_to_buffer(state)
        
    # Benchmark
    N_ITERS = 1000
    print(f"Benchmarking state_to_buffer ({N_ITERS} iterations)...")
    
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = state_to_buffer(state)
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    avg_time_ms = (total_time / N_ITERS) * 1000
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per call: {avg_time_ms:.4f}ms")

if __name__ == "__main__":
    benchmark()
