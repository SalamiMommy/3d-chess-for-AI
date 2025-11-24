#!/usr/bin/env python3
"""Test script to verify the freeze expiry bug fix."""

import numpy as np
from game3d.game.factory import start_game_state
from game3d.movement.generator import generate_legal_moves

def test_move_generation_not_frozen():
    """Verify that pieces are not incorrectly marked as frozen at game start."""
    print("Testing move generation fix...")
    
    # Start a new game
    state = start_game_state()
    
    # Generate legal moves for White (turn 0)
    moves = generate_legal_moves(state)
    
    # Basic sanity check
    num_moves = len(moves) if moves is not None else 0
    num_pieces = len(state.cache_manager.occupancy_cache.get_positions(state.color))
    
    print(f"Turn: {state.turn_number}")
    print(f"Active color: {'WHITE' if state.color == 1 else 'BLACK'}")
    print(f"Number of pieces: {num_pieces}")
    print(f"Number of legal moves: {num_moves}")
    
    # Verify we have legal moves
    if num_moves == 0:
        print("❌ FAILED: No legal moves generated (pieces are frozen)")
        return False
    
    if num_moves > 0:
        print(f"✅ SUCCESS: {num_moves} legal moves generated")
        print(f"Sample moves: {moves[:5]}")
        return True
    
    return False

if __name__ == "__main__":
    success = test_move_generation_not_frozen()
    exit(0 if success else 1)
