"""Quick benchmark to test parallelization improvements."""
import time
import numpy as np
from game3d.game.factory import start_game_state
from game3d.movement.generator import generate_legal_moves

def benchmark_move_generation(num_iterations=50):
    """Benchmark legal move generation."""
    print(f"Benchmarking move generation ({num_iterations} iterations)...")
    
    # Initialize game
    state = start_game_state()
    
    # Warm-up (compile numba functions)
    print("Warming up...")
    for _ in range(3):
        moves = generate_legal_moves(state)
    
    # Actual benchmark
    print(f"\nRunning benchmark...")
    start = time.perf_counter()
    
    for i in range(num_iterations):
        moves = generate_legal_moves(state)
        if i == 0:
            print(f"  First iteration: {len(moves)} legal moves generated")
    
    end = time.perf_counter()
    
    total_time = end - start
    avg_time = total_time / num_iterations
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time per iteration: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {num_iterations/total_time:.1f} iterations/sec")
    
    return avg_time

if __name__ == "__main__":
    benchmark_move_generation()
