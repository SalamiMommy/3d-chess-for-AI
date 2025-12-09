"""
Profile generator performance: Legacy vs New Stateless.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import numpy as np

def main():
    print("=== Generator Performance Profile ===\n")
    
    # Load game state
    print("Loading game state...")
    from game3d.game.gamestate import GameState
    state = GameState.from_startpos()
    
    # Convert to buffer
    print("Converting to buffer...")
    from game3d.core.buffer import state_to_buffer
    buffer = state_to_buffer(state)
    print(f"  Occupied count: {buffer.occupied_count}")
    print(f"  Active color: {buffer.meta[0]}")
    
    # Import systems
    print("\nImporting move generation systems...")
    from game3d.movement.generator import generate_legal_moves as legacy_gen
    from game3d.core.api import generate_legal_moves as new_gen, invalidate_cache
    from game3d.core.generator_functional import generate_moves as new_pseudo
    from game3d.core.attacks import filter_legal_moves as new_filter
    
    # Warmup (JIT compilation)
    print("\nWarming up (JIT compile)...")
    _ = legacy_gen(state)
    _ = new_gen(buffer)
    invalidate_cache()
    
    # Benchmark
    N_ITERS = 20
    
    print(f"\nBenchmarking ({N_ITERS} iterations)...\n")
    
    # 1. Legacy (cached)
    legacy_moves = legacy_gen(state)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = legacy_gen(state)
    t_legacy_cached = (time.perf_counter() - t0) / N_ITERS * 1000
    
    # 2. New Stateless (cached)
    new_moves = new_gen(buffer)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = new_gen(buffer, use_cache=True)
    t_new_cached = (time.perf_counter() - t0) / N_ITERS * 1000
    
    # 3. New Stateless (fresh generation each time)
    invalidate_cache()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        invalidate_cache()
        _ = new_gen(buffer, use_cache=True)
    t_new_fresh = (time.perf_counter() - t0) / N_ITERS * 1000
    
    # 4. New Stateless - pseudo-legal only
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = new_pseudo(buffer)
    t_new_pseudo = (time.perf_counter() - t0) / N_ITERS * 1000
    
    # 5. New Stateless - filtering only (from pseudo-legal)
    pseudo = new_pseudo(buffer)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = new_filter(buffer, pseudo)
    t_new_filter = (time.perf_counter() - t0) / N_ITERS * 1000
    
    # Results
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"\nLegacy (cached):        {t_legacy_cached:.2f} ms")
    print(f"New (cached):           {t_new_cached:.4f} ms (speedup: {t_legacy_cached/t_new_cached:.0f}x)" if t_new_cached > 0 else "New (cached): ~0 ms")
    print(f"New (fresh):            {t_new_fresh:.2f} ms")
    print(f"New (pseudo-legal):     {t_new_pseudo:.2f} ms")
    print(f"New (filter only):      {t_new_filter:.2f} ms")
    
    print(f"\nMove counts:")
    print(f"  Legacy: {len(legacy_moves)} moves")
    print(f"  New:    {len(new_moves)} moves")
    
    if len(legacy_moves) != len(new_moves):
        print(f"\n⚠ MISMATCH: {len(new_moves) - len(legacy_moves):+d} moves difference")
    else:
        print(f"\n✓ Move counts match!")

if __name__ == "__main__":
    main()
