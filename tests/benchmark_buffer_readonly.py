
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
        _ = state_to_buffer(state, readonly=True)
        
    # Benchmark
    N_ITERS = 1000
    # Benchmark Readonly=False
    print(f"Benchmarking state_to_buffer(readonly=False) ({N_ITERS} iterations)...")
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = state_to_buffer(state, readonly=False)
    t1 = time.perf_counter()
    print(f"False: {(t1-t0)*1000/N_ITERS:.4f} ms")

    # Benchmark Readonly=True
    print(f"Benchmarking state_to_buffer(readonly=True) ({N_ITERS} iterations)...")
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = state_to_buffer(state, readonly=True)
    t1 = time.perf_counter()
    print(f"True:  {(t1-t0)*1000/N_ITERS:.4f} ms")

if __name__ == "__main__":
    benchmark()
