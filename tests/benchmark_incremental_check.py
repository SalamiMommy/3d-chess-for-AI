#!/usr/bin/env python3
"""
Benchmark incremental check detection optimization.

Compares performance of:
1. Old path: _square_attacked_by_slow (regenerates all opponent moves)
2. New path: square_attacked_by_incremental (only regenerates affected pieces)

Expected speedup: 10-20x
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from game3d.game.factory import create_board
from game3d.cache.manager import CacheManager
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType
from game3d.attacks.check import square_attacked_by_incremental, _square_attacked_by_slow
from game3d.movement.generator import generate_legal_moves

def setup_game_state():
    """Create a game state with some pieces for testing."""
    board = create_board()
    cache = CacheManager(board)
    
    # Initialize with a standard starting position
    from game3d.game.terminal import setup_default_board
    setup_default_board(board, cache)
    
    state = GameState(board, Color.WHITE, cache)
    
    # Generate legal moves to populate cache
    _ = generate_legal_moves(state)
    
    # Switch to black to populate their cache too
    state_black = GameState(board, Color.BLACK, cache)
    _ = generate_legal_moves(state_black)
    
    return board, cache, state

def benchmark_check_detection(num_trials=100):
    """Benchmark check detection performance."""
    print("=" * 70)
    print("INCREMENTAL CHECK DETECTION BENCHMARK")
    print("=" * 70)
    
    # Setup
    print("\nSetting up game state...")
    board, cache, state = setup_game_state()
    occ_cache = cache.occupancy_cache
    
    # Find a valid move to simulate
    white_positions = occ_cache.get_positions(Color.WHITE)
    if len(white_positions) == 0:
        print("ERROR: No white pieces found")
        return
    
    # Use first white piece for testing
    from_coord = white_positions[0]
    
    # Find a valid destination (just move one square in any direction)
    to_coord = from_coord.copy()
    to_coord[0] = min(7, to_coord[0] + 1)  # Move one square in x direction
    
    # Find king position to check
    king_pos = occ_cache.find_king(Color.WHITE)
    if king_pos is None:
        print("ERROR: King not found")
        return
    
    print(f"\nTest configuration:")
    print(f"  - From: {from_coord}")
    print(f"  - To: {to_coord}")
    print(f"  - King position: {king_pos}")
    print(f"  - Trials: {num_trials}")
    
    # Get piece info
    piece_data = occ_cache.get(from_coord)
    if not piece_data:
        print("ERROR: No piece at from_coord")
        return
    
    # Simulate the move
    occ_cache.set_position(from_coord, None)
    occ_cache.set_position(to_coord, np.array([piece_data['piece_type'], piece_data['color']]))
    
    try:
        # Benchmark 1: Incremental path (NEW)
        print("\n" + "-" * 70)
        print("Testing INCREMENTAL path (new optimization)...")
        print("-" * 70)
        
        times_incremental = []
        for i in range(num_trials):
            start = time.perf_counter()
            result_incremental = square_attacked_by_incremental(
                board,
                king_pos,
                Color.BLACK,  # Check if black attacks king
                cache,
                from_coord,
                to_coord
            )
            end = time.perf_counter()
            times_incremental.append((end - start) * 1000000)  # Convert to microseconds
        
        avg_incr = np.mean(times_incremental)
        std_incr = np.std(times_incremental)
        print(f"Result: {result_incremental}")
        print(f"Average time: {avg_incr:.2f} μs")
        print(f"Std dev: {std_incr:.2f} μs")
        print(f"Min: {np.min(times_incremental):.2f} μs")
        print(f"Max: {np.max(times_incremental):.2f} μs")
        
        # Benchmark 2: Slow path (OLD)
        print("\n" + "-" * 70)
        print("Testing SLOW path (old full regeneration)...")
        print("-" * 70)
        
        times_slow = []
        for i in range(num_trials):
            start = time.perf_counter()
            result_slow = _square_attacked_by_slow(
                board,
                king_pos,
                Color.BLACK,
                cache
            )
            end = time.perf_counter()
            times_slow.append((end - start) * 1000000)  # Convert to microseconds
        
        avg_slow = np.mean(times_slow)
        std_slow = np.std(times_slow)
        print(f"Result: {result_slow}")
        print(f"Average time: {avg_slow:.2f} μs")
        print(f"Std dev: {std_slow:.2f} μs")
        print(f"Min: {np.min(times_slow):.2f} μs")
        print(f"Max: {np.max(times_slow):.2f} μs")
        
        # Results comparison
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        if result_incremental != result_slow:
            print("⚠️  WARNING: Results differ!")
            print(f"   Incremental: {result_incremental}")
            print(f"   Slow: {result_slow}")
        else:
            print("✅ Results match: Both methods agree")
        
        speedup = avg_slow / avg_incr
        print(f"\n📊 Performance Improvement:")
        print(f"   Old (slow) path:     {avg_slow:.2f} μs")
        print(f"   New (incremental):   {avg_incr:.2f} μs")
        print(f"   Speedup:             {speedup:.2f}x faster")
        
        if speedup >= 10:
            print(f"   ✅ TARGET MET: {speedup:.2f}x speedup (target: 10-20x)")
        elif speedup >= 5:
            print(f"   ⚠️  Good but below target: {speedup:.2f}x (target: 10-20x)")
        else:
            print(f"   ❌ Below expectations: {speedup:.2f}x (target: 10-20x)")
        
    finally:
        # Revert the move
        occ_cache.set_position(to_coord, None)
        occ_cache.set_position(from_coord, np.array([piece_data['piece_type'], piece_data['color']]))

if __name__ == "__main__":
    benchmark_check_detection(num_trials=100)
