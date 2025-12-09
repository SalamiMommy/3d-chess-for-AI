
import sys
import os
import time
import numpy as np
from game3d.game.gamestate import GameState
from game3d.core.buffer import state_to_buffer

def debug_perf():
    state = GameState.from_startpos()
    cache = state.cache_manager.occupancy_cache
    occ_grid = cache._occ
    
    # Test copy vs ref
    N = 5000
    
    t0 = time.perf_counter()
    for _ in range(N):
        x = occ_grid.copy()
        y = x.flatten(order='F')
    t1 = time.perf_counter()
    print(f"Copy + Flatten: {(t1-t0)*1000/N:.4f} ms")
    
    t0 = time.perf_counter()
    for _ in range(N):
        x = occ_grid
        y = x.flatten(order='F')
    t1 = time.perf_counter()
    print(f"Ref + Flatten: {(t1-t0)*1000/N:.4f} ms")

if __name__ == "__main__":
    debug_perf()
