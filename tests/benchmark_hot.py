
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.getcwd()))

from game3d.game.gamestate import GameState
from game3d.core.buffer import state_to_buffer
from game3d.board.symmetry import SymmetryManager
from game3d.common.coord_utils import coords_to_keys

def benchmark():
    print("Initializing GameState...")
    state = GameState.from_startpos()
    
    print("Initializing SymmetryManager...")
    sym_mgr = SymmetryManager()
    
    # Create some dummy coords for coords_to_keys
    coords = np.random.randint(0, 8, size=(100, 3)).astype(np.int16)
    
    # Warmup
    print("Warming up...")
    _ = state_to_buffer(state)
    _ = sym_mgr.get_canonical_form(state.board.array())
    _ = coords_to_keys(coords)
        
    # Benchmark state_to_buffer
    N_ITERS = 1000
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = state_to_buffer(state)
    t_buffer = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"state_to_buffer: {t_buffer:.4f} ms")

    # Benchmark get_canonical_form
    # Use array directly as it's the expected input type for optimization (usually)
    # But let's follow normal usage pattern if possible. state.board.array() is typical
    board_arr = state.board.array()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = sym_mgr.get_canonical_form(board_arr)
    t_canonical = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"get_canonical_form (Dense): {t_canonical:.4f} ms")

    # Benchmark get_canonical_form (Sparse/Cached)
    # Using state.board triggers the fast path
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = sym_mgr.get_canonical_form(state.board)
    t_canonical_sparse = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"get_canonical_form (Sparse): {t_canonical_sparse:.4f} ms")


    # Benchmark coords_to_keys
    # Use a larger batch for realistic load
    large_coords = np.random.randint(0, 8, size=(1000, 3)).astype(np.int16)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = coords_to_keys(large_coords)
    t_keys = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"coords_to_keys (N=1000): {t_keys:.4f} ms")

if __name__ == "__main__":
    benchmark()
