"""
Performance benchmark for check detection and move generation.
Run before and after optimizations to measure improvement.
"""
import time
import numpy as np
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color

def benchmark_move_generation(iterations=100):
    """Benchmark the full move generation pipeline."""
    state = GameState(Board.startpos(), Color.WHITE)
    
    # Warm up Numba JIT
    print("Warming up JIT...")
    for _ in range(5):
        state.gen_moves()
        state.cache_manager.move_cache.invalidate()
    
    # Benchmark with cache invalidation (cold cache)
    print(f"\nBenchmarking {iterations} iterations (cold cache)...")
    start = time.perf_counter()
    for _ in range(iterations):
        state.cache_manager.move_cache.invalidate()
        state.gen_moves()
    cold_elapsed = time.perf_counter() - start
    
    print(f"Cold cache: {cold_elapsed:.2f}s ({cold_elapsed/iterations*1000:.1f}ms avg)")
    
    # Benchmark with warm cache (incremental)
    print(f"\nBenchmarking {iterations} iterations (warm cache)...")
    start = time.perf_counter()
    for _ in range(iterations):
        state.gen_moves()
    warm_elapsed = time.perf_counter() - start
    
    print(f"Warm cache: {warm_elapsed:.2f}s ({warm_elapsed/iterations*1000:.1f}ms avg)")
    
    return cold_elapsed, warm_elapsed

def benchmark_check_detection(iterations=1000):
    """Benchmark just the check detection portion."""
    from game3d.attacks.check import move_would_leave_king_in_check
    
    state = GameState(Board.startpos(), Color.WHITE)
    moves = state.gen_moves()
    
    if len(moves) == 0:
        print("No moves to benchmark!")
        return
    
    # Warm up
    print("\nWarming up check detection...")
    for i in range(min(10, len(moves))):
        move_would_leave_king_in_check(state, moves[i], state.cache_manager)
    
    # Benchmark
    print(f"\nBenchmarking {iterations} check detections...")
    start = time.perf_counter()
    for i in range(iterations):
        move_would_leave_king_in_check(state, moves[i % len(moves)], state.cache_manager)
    elapsed = time.perf_counter() - start
    
    print(f"Check detection: {elapsed:.2f}s ({elapsed/iterations*1000:.3f}ms avg)")
    
    return elapsed

if __name__ == "__main__":
    print("=" * 60)
    print("3D Chess Performance Benchmark")
    print("=" * 60)
    
    benchmark_move_generation()
    benchmark_check_detection()
    
    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)
