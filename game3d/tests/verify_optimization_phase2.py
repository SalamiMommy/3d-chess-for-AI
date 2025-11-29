#!/usr/bin/env python3
"""Verification script for Phase 2 optimizations."""

import numpy as np
import sys
import time
import timeit
import cProfile
import pstats
from io import StringIO

sys.path.insert(0, '/home/salamimommy/Documents/code/3d-chess-for-AI')

from game3d.common.shared_types import COORD_DTYPE, SIZE, Color, PieceType
from game3d.common.coord_utils import in_bounds_vectorized
from game3d.pieces.pieces.friendlytp import generate_friendlytp_moves
from game3d.cache.manager import OptimizedCacheManager
from game3d.cache.caches.occupancycache import OccupancyCache

def benchmark_friendlytp():
    print("\n=== Benchmarking FriendlyTP ===")
    
    class MockBoard:
        def __init__(self):
            self.size = SIZE
        def get_initial_setup(self):
            return (np.empty((0, 3), dtype=COORD_DTYPE), np.empty(0, dtype=int), np.empty(0, dtype=int))
            
    cm = OptimizedCacheManager(MockBoard())
    
    # Setup: Place many friendly pieces to create a complex network
    # Create a grid of friendly pieces
    coords = []
    for x in range(0, SIZE, 2):
        for y in range(0, SIZE, 2):
            for z in range(0, SIZE, 2):
                coords.append([x, y, z])
    
    coords = np.array(coords, dtype=COORD_DTYPE)
    pieces = np.zeros((len(coords), 2), dtype=int)
    pieces[:, 0] = PieceType.PAWN.value
    pieces[:, 1] = Color.WHITE.value
    
    cm.occupancy_cache.batch_set_positions(coords, pieces)
    
    # Test position
    start_pos = np.array([1, 1, 1], dtype=COORD_DTYPE)
    
    # Warmup
    for _ in range(10):
        generate_friendlytp_moves(cm, Color.WHITE, start_pos)
        
    # Benchmark
    t0 = timeit.default_timer()
    n_iters = 100
    for _ in range(n_iters):
        generate_friendlytp_moves(cm, Color.WHITE, start_pos)
    t1 = timeit.default_timer()
    
    print(f"FriendlyTP generation ({n_iters} iters): {t1 - t0:.4f}s")
    print(f"Average time: {(t1 - t0) / n_iters * 1000:.4f}ms")

def benchmark_occupancy_set_positions():
    print("\n=== Benchmarking OccupancyCache.batch_set_positions ===")
    
    occ = OccupancyCache()
    
    # Generate random updates
    N = 1000
    coords = np.random.randint(0, SIZE, size=(N, 3)).astype(COORD_DTYPE)
    pieces = np.random.randint(0, 10, size=(N, 2)).astype(int)
    
    # Benchmark
    t0 = timeit.default_timer()
    n_iters = 100
    for _ in range(n_iters):
        occ.batch_set_positions(coords, pieces)
    t1 = timeit.default_timer()
    
    print(f"batch_set_positions ({n_iters} iters, N={N}): {t1 - t0:.4f}s")
    print(f"Average time: {(t1 - t0) / n_iters * 1000:.4f}ms")

def benchmark_in_bounds():
    print("\n=== Benchmarking in_bounds ===")
    
    # Test different batch sizes
    batch_sizes = [1, 3, 10, 100, 1000]
    
    for N in batch_sizes:
        coords = np.random.randint(-1, SIZE + 1, size=(N, 3)).astype(COORD_DTYPE)
        
        t0 = timeit.default_timer()
        n_iters = 10000 if N < 100 else 1000
        for _ in range(n_iters):
            in_bounds_vectorized(coords)
        t1 = timeit.default_timer()
        
        print(f"in_bounds (N={N}): {(t1 - t0) / n_iters * 1000:.4f}ms per call")

if __name__ == "__main__":
    benchmark_friendlytp()
    benchmark_occupancy_set_positions()
    benchmark_in_bounds()
