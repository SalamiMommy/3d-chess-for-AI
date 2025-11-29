#!/usr/bin/env python3
"""Verification script for move generation optimizations."""

import numpy as np
import sys
import time
import timeit
sys.path.insert(0, '/home/salamimommy/Documents/code/3d-chess-for-AI')

from game3d.common.shared_types import COORD_DTYPE, SIZE, Color, PieceType, PAWN_START_RANK_WHITE
from game3d.common.coord_utils import in_bounds_vectorized
from game3d.pieces.pieces.wall import generate_wall_moves
from game3d.pieces.pieces.pawn import generate_pawn_moves
from game3d.cache.manager import OptimizedCacheManager
from game3d.game.gamestate import GameState

def test_wall_moves():
    print("\n=== Testing Wall Moves ===")
    
    # Setup
    class MockBoard:
        def __init__(self):
            self.size = SIZE
        def get_initial_setup(self):
            return (np.empty((0, 3), dtype=COORD_DTYPE), np.empty(0, dtype=int), np.empty(0, dtype=int))
            
    cm = OptimizedCacheManager(MockBoard())
    
    # 1. Test Center Wall (should have 6 moves if empty)
    center_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    moves = generate_wall_moves(cm, Color.WHITE, center_pos)
    print(f"Center Wall Moves: {len(moves)}")
    if len(moves) != 6:
        print(f"❌ Expected 6 moves, got {len(moves)}")
    else:
        print("✅ Center Wall count correct")
        
    # 2. Test Corner Wall (should have fewer moves)
    corner_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    moves = generate_wall_moves(cm, Color.WHITE, corner_pos)
    print(f"Corner Wall Moves: {len(moves)}")
    # Directions: +x, +y, +z. -x, -y, -z are out of bounds.
    # But wait, wall is 2x2x1.
    # Anchor at 0,0,0 occupies (0,0,0), (1,0,0), (0,1,0), (1,1,0)
    # Move +x: Anchor becomes 1,0,0. Occupies (1,0,0), (2,0,0), (1,1,0), (2,1,0). Valid.
    # Move +y: Anchor becomes 0,1,0. Occupies (0,1,0), (1,1,0), (0,2,0), (1,2,0). Valid.
    # Move +z: Anchor becomes 0,0,1. Occupies (0,0,1), (1,0,1), (0,1,1), (1,1,1). Valid.
    # Move -x: Anchor -1,0,0. Invalid.
    # Move -y: Anchor 0,-1,0. Invalid.
    # Move -z: Anchor 0,0,-1. Invalid.
    # So expected 3 moves.
    if len(moves) != 3:
        print(f"❌ Expected 3 moves, got {len(moves)}")
    else:
        print("✅ Corner Wall count correct")

    # 3. Test Blocked Wall
    # Place a friendly piece at 5,4,4 (blocking +x move for center wall at 4,4,4)
    # Center wall at 4,4,4 occupies (4,4,4), (5,4,4), (4,5,4), (5,5,4)
    # Move +x: Anchor 5,4,4. Occupies (5,4,4), (6,4,4), (5,5,4), (6,5,4)
    # If we place a friendly piece at 6,4,4, it should block the move.
    
    cm.occupancy_cache.set_position(np.array([6, 4, 4], dtype=COORD_DTYPE), np.array([PieceType.PAWN, Color.WHITE]))
    moves = generate_wall_moves(cm, Color.WHITE, center_pos)
    print(f"Blocked Wall Moves: {len(moves)}")
    # Should be 5 moves (all except +x)
    if len(moves) != 5:
        print(f"❌ Expected 5 moves, got {len(moves)}")
        # Check which move is missing/present
        # +x move is to anchor 5,4,4
        has_plus_x = any(np.array_equal(m[3:], [5, 4, 4]) for m in moves)
        if has_plus_x:
            print("  ❌ +x move was generated but should be blocked")
    else:
        print("✅ Blocked Wall count correct")
        
    # Clean up
    cm.occupancy_cache.set_position(np.array([6, 4, 4], dtype=COORD_DTYPE), None)


def test_pawn_moves_batch():
    print("\n=== Testing Pawn Moves (Batch) ===")
    # This test will be more relevant once we implement batch generation
    # For now, we just verify the single pawn generation still works
    
    class MockBoard:
        def __init__(self):
            self.size = SIZE
        def get_initial_setup(self):
            return (np.empty((0, 3), dtype=COORD_DTYPE), np.empty(0, dtype=int), np.empty(0, dtype=int))
    
    cm = OptimizedCacheManager(MockBoard())
    
    # White pawn at start rank
    y_start = PAWN_START_RANK_WHITE
    pos = np.array([4, y_start, 4], dtype=COORD_DTYPE)
    moves = generate_pawn_moves(cm, Color.WHITE, pos)
    print(f"Start Rank Pawn Moves: {len(moves)}")
    # Should have single push and double push (if empty)
    if len(moves) != 2:
        print(f"❌ Expected 2 moves, got {len(moves)}")
    else:
        print("✅ Start Rank Pawn count correct")
        
    # Place enemy for capture
    # White pawn attacks +1, +1, +1 and -1, +1, +1 etc.
    # Attack square: 5, y_start + 1, 5
    cm.occupancy_cache.set_position(np.array([5, y_start + 1, 5], dtype=COORD_DTYPE), np.array([PieceType.PAWN, Color.BLACK]))
    moves = generate_pawn_moves(cm, Color.WHITE, pos)
    print(f"Capturing Pawn Moves: {len(moves)}")
    # 2 pushes + 1 capture = 3
    if len(moves) != 3:
        print(f"❌ Expected 3 moves, got {len(moves)}")
    else:
        print("✅ Capturing Pawn count correct")


def benchmark_in_bounds():
    print("\n=== Benchmarking in_bounds ===")
    
    # Generate random coordinates
    N = 10000
    coords = np.random.randint(-1, SIZE + 1, size=(N, 3)).astype(COORD_DTYPE)
    
    # Benchmark current implementation
    t0 = timeit.default_timer()
    for _ in range(100):
        _ = in_bounds_vectorized(coords)
    t1 = timeit.default_timer()
    print(f"Current in_bounds (100 iters, N={N}): {t1 - t0:.4f}s")
    
    # Benchmark simple numpy implementation
    def in_bounds_simple(c):
        return np.all((c >= 0) & (c < SIZE), axis=1)
        
    t0 = timeit.default_timer()
    for _ in range(100):
        _ = in_bounds_simple(coords)
    t1 = timeit.default_timer()
    print(f"Simple numpy in_bounds (100 iters, N={N}): {t1 - t0:.4f}s")

    # Benchmark Numba parallel implementation
    from numba import njit, prange
    
    @njit(parallel=True, fastmath=True)
    def in_bounds_numba_parallel(coords):
        n = coords.shape[0]
        res = np.empty(n, dtype=np.bool_)
        for i in prange(n):
            res[i] = (coords[i, 0] >= 0) & (coords[i, 0] < SIZE) & \
                     (coords[i, 1] >= 0) & (coords[i, 1] < SIZE) & \
                     (coords[i, 2] >= 0) & (coords[i, 2] < SIZE)
        return res

    # Compile first
    _ = in_bounds_numba_parallel(coords[:10])
    
    t0 = timeit.default_timer()
    for _ in range(100):
        _ = in_bounds_numba_parallel(coords)
    t1 = timeit.default_timer()
    print(f"Numba parallel in_bounds (100 iters, N={N}): {t1 - t0:.4f}s")


if __name__ == "__main__":
    test_wall_moves()
    test_pawn_moves_batch()
    benchmark_in_bounds()
