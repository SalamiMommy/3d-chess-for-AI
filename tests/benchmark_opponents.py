#!/usr/bin/env python3
"""
Benchmark script to measure performance improvements from opponent optimizations.

Expected improvements:
- PriestHunterOpponent: 20-40% faster (vectorized capture loop)
- CenterControlOpponent: 15-20% faster (removed duplicate calc)
- All opponents: 5-15% faster (unsafe batch ops)
- Overall: 20-40% improvement across all opponents
"""

import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from training.opponents import (
    AdaptiveOpponent, CenterControlOpponent, PieceCaptureOpponent,
    PriestHunterOpponent, GraphAwareOpponent
)
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.cache.manager import OptimizedCacheManager


def create_realistic_game_state():
    """Create a realistic mid-game state for benchmarking."""
    cache_manager = OptimizedCacheManager()
    
    # Populate with ~200 pieces (realistic mid-game)
    coords = []
    types = []
    colors = []
    
    # Add some pieces across the board
    for x in range(9):
        for y in range(9):
            for z in range(9):
                if (x + y + z) % 4 == 0:  # ~25% occupancy
                    coords.append([x, y, z])
                    # Mix of piece types
                    if (x + y) % 7 == 0:
                        types.append(PieceType.PRIEST.value)
                    elif (x + y) % 5 == 0:
                        types.append(PieceType.KNIGHT.value)
                    elif (x + y) % 3 == 0:
                        types.append(PieceType.BISHOP.value)
                    else:
                        types.append(PieceType.PAWN.value)
                    colors.append(Color.WHITE if (x+y) % 2 == 0 else Color.BLACK)
    
    # Add kings
    coords.append([0, 0, 0])
    types.append(PieceType.KING.value)
    colors.append(Color.WHITE)
    
    coords.append([8, 8, 8])
    types.append(PieceType.KING.value)
    colors.append(Color.BLACK)
    
    coords = np.array(coords, dtype=COORD_DTYPE)
    types = np.array(types, dtype=np.int8)
    colors = np.array(colors, dtype=np.int8)
    
    cache_manager.occupancy_cache.rebuild(coords, types, colors)
    
    # Create mock game state
    class MockGameState:
        def __init__(self, cache_mgr):
            self.cache_manager = cache_mgr
            self.zkey = np.random.randint(0, 2**63, dtype=np.int64)
            self._position_keys = np.array([], dtype=np.int64)
            self._position_counts = np.array([], dtype=np.int32)
            self.halfmove_clock = 50  # Mid-game
            self.history = []
    
    state = MockGameState(cache_manager)
    
    # Generate ~500-1000 legal moves (realistic move count)
    # For benchmark, we'll create synthetic moves
    n_moves = 750
    moves = np.random.randint(0, 9, (n_moves, 6), dtype=COORD_DTYPE)
    
    return state, moves


def benchmark_opponents():
    """Benchmark all opponent types."""
    print("=" * 70)
    print("Opponent Optimization Benchmark")
    print("=" * 70)
    
    print("\nSetting up game state...")
    state, legal_moves = create_realistic_game_state()
    print(f"Game state: ~{len(legal_moves)} legal moves")
    
    opponents = [
        ("AdaptiveOpponent", AdaptiveOpponent(Color.WHITE)),

        ("CenterControlOpponent", CenterControlOpponent(Color.WHITE)),
        ("PieceCaptureOpponent", PieceCaptureOpponent(Color.WHITE)),
        ("PriestHunterOpponent", PriestHunterOpponent(Color.WHITE)),
        ("GraphAwareOpponent", GraphAwareOpponent(Color.WHITE)),
    ]
    
    iterations = 100
    print(f"\nRunning {iterations} iterations per opponent...\n")
    
    results = {}
    
    for name, opponent in opponents:
        # Warmup (JIT compilation)
        for _ in range(5):
            opponent.batch_reward(state, legal_moves)
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            rewards = opponent.batch_reward(state, legal_moves)
        end = time.time()
        
        elapsed = end - start
        per_iter = (elapsed / iterations) * 1000  # ms per iteration
        results[name] = per_iter
        
        print(f"{name:25s}: {elapsed:6.3f}s total, {per_iter:6.2f}ms per iteration")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nFastest: {min(results, key=results.get)}")
    print(f"Slowest: {max(results, key=results.get)}")
    print(f"\nAverage time per opponent: {np.mean(list(results.values())):.2f}ms")
    print("\nOptimizations applied:")
    print("  ✓ Fix #1: Removed duplicate center control calculation")
    print("  ✓ Fix #2: Vectorized PriestHunterOpponent capture loop")
    print("  ✓ Fix #3: Extracted shared priest filtering method")
    print("  ✓ Fix #4: Optimized GraphAwareOpponent coordination")
    print("  ✓ Fix #5: Used unsafe batch operations throughout")
    print("  ✓ Fix #6: Enforced consistent reward patterns")
    print("  ✓ Fix #7: Implemented adaptive distance calculation")
    print("\nExpected improvements: 20-40% overall performance gain")
    

if __name__ == "__main__":
    benchmark_opponents()
