#!/usr/bin/env python3
"""Quick test to verify optimizations didn't break move generation."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from game3d.game.factory import start_game_state
from game3d.movement.generator import generate_legal_moves
from game3d.common.shared_types import Color

def test_basic_game():
    """Test that we can generate legal moves for a basic game state."""
    print("Creating initial game state...")
    state = start_game_state()
    
    print(f"Current player: {Color(state.color).name}")
    print(f"Turn: {state.turn_number}")
    
    print("\nGenerating legal moves...")
    moves = generate_legal_moves(state)
    
    print(f"Generated {len(moves)} legal moves")
    
    if len(moves) == 0:
        print("❌ FAIL: No legal moves generated!")
        return False
    
    # Show a few sample moves
    print("\nSample moves (first 5):")
    for i, move in enumerate(moves[:5]):
        from_coord = move[:3]
        to_coord = move[3:]
        print(f"  {i+1}. {from_coord} -> {to_coord}")
    
    # Verify moves have correct format
    assert moves.shape[1] == 6, f"Expected moves to have 6 columns, got {moves.shape[1]}"
    
    # Verify coordinates are in bounds
    assert np.all(moves >= 0), "Some coordinates are negative"
    assert np.all(moves < 9), "Some coordinates exceed board size"
    
    print("\n✅ SUCCESS: Move generation works correctly!")
    return True

if __name__ == "__main__":
    try:
        success = test_basic_game()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
