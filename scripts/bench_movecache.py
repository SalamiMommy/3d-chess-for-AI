#!/usr/bin/env python
"""Benchmark script for move cache optimizations."""

import sys
import time
import numpy as np

# Add project root to path
sys.path.insert(0, '.')

def benchmark_move_generation(num_games=5, moves_per_game=10):
    """Benchmark move generation with the new batch optimizations."""
    from game3d.game.gamestate import GameState
    from game3d.board.board import Board
    from game3d.movement.generator import generate_legal_moves
    
    times = []
    move_counts = []
    
    for game_idx in range(num_games):
        board = Board()
        state = GameState(board)  # Create with board
        
        game_start = time.perf_counter()
        total_moves_generated = 0
        
        for move_idx in range(moves_per_game):
            # Invalidate cache to force fresh generation
            state.cache_manager.move_cache.invalidate_legal_moves(state.color)
            state.cache_manager.move_cache.invalidate_pseudolegal_moves(state.color)
            
            # Generate legal moves
            start = time.perf_counter()
            moves = generate_legal_moves(state)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            total_moves_generated += len(moves)
            
            if len(moves) == 0:
                print(f"  Game {game_idx+1}: No moves at iteration {move_idx}")
                break
        
        game_elapsed = time.perf_counter() - game_start
        move_counts.append(total_moves_generated)
        print(f"Game {game_idx+1}: {total_moves_generated} moves in {game_elapsed:.3f}s ({moves_per_game} iterations)")
    
    times = np.array(times)
    print(f"\n=== Results ===")
    print(f"Total generations: {len(times)}")
    print(f"Mean time: {np.mean(times)*1000:.2f}ms")
    print(f"Median time: {np.median(times)*1000:.2f}ms")
    print(f"Std dev: {np.std(times)*1000:.2f}ms")
    print(f"Min/Max: {np.min(times)*1000:.2f}ms / {np.max(times)*1000:.2f}ms")
    print(f"Total moves: {sum(move_counts)}")


def test_batch_functions():
    """Quick sanity test of the new batch functions."""
    print("Testing batch Numba functions...")
    
    from game3d.cache.caches.movecache import (
        _find_group_boundaries,
        _unique_per_piece_batch,
        _compute_batch_bit_ops
    )
    
    # Test _find_group_boundaries
    sorted_keys = np.array([1, 1, 2, 2, 2, 5, 5, 10], dtype=np.int64)
    unique, starts, ends = _find_group_boundaries(sorted_keys)
    
    assert len(unique) == 4, f"Expected 4 groups, got {len(unique)}"
    assert list(unique) == [1, 2, 5, 10], f"Wrong unique keys: {unique}"
    assert list(starts) == [0, 2, 5, 7], f"Wrong starts: {starts}"
    assert list(ends) == [2, 5, 7, 8], f"Wrong ends: {ends}"
    print("  _find_group_boundaries: PASS")
    
    # Test _unique_per_piece_batch
    all_keys = np.array([3, 1, 2, 1, 5, 5, 3], dtype=np.int64)
    starts = np.array([0, 3], dtype=np.int32)
    ends = np.array([3, 7], dtype=np.int32)
    
    unique_keys, offsets, counts = _unique_per_piece_batch(all_keys, starts, ends, 2)
    
    # First piece has [3,1,2] -> sorted unique [1,2,3]
    # Second piece has [1,5,5,3] -> sorted unique [1,3,5]
    assert counts[0] == 3, f"Expected 3 unique for piece 0, got {counts[0]}"
    assert counts[1] == 3, f"Expected 3 unique for piece 1, got {counts[1]}"
    print("  _unique_per_piece_batch: PASS")
    
    print("All batch function tests passed!")


if __name__ == "__main__":
    # First run quick function tests
    test_batch_functions()
    print()
    
    # Then run benchmark
    print("Running move generation benchmark...")
    print("(This will warm up Numba JIT on first runs)")
    print()
    benchmark_move_generation(num_games=3, moves_per_game=5)
