#!/usr/bin/env python3
"""Test script to verify Wall OOB fix."""

import numpy as np
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import PieceType, Color
from game3d.game.turnmove import make_move

def test_wall_oob_prevention():
    """Test that Wall moves to invalid positions are properly rejected."""
    print("Testing Wall OOB Prevention...")
    
    board = Board.startpos()
    game = GameState(board=board)
    
    # Clear the board first
    for color in [Color.WHITE, Color.BLACK]:
        all_coords = game.cache_manager.occupancy_cache.get_positions(color)
        for coord in all_coords:
            game.cache_manager.occupancy_cache.set_position(coord, None)
   
    # Set game to White's turn
    game.color = Color.WHITE
    
    # Place Wall at [7, 7, 5] (Valid anchor: 7,7,5; 8,7,5; 7,8,5; 8,8,5)
    wall_pos = np.array([7, 7, 5])
    game.cache_manager.occupancy_cache.set_position(
        wall_pos, 
        np.array([PieceType.WALL, Color.WHITE])
    )
    
    # Attempt to move Wall to [8, 7, 5] (INVALID - would create [9,7,5])
    invalid_move = np.array([7, 7, 5, 8, 7, 5], dtype=np.int16)
    
    print(f"Attempting invalid Wall move: {wall_pos} -> {invalid_move[3:]}")
    
    try:
        new_state = make_move(game, invalid_move)
        print("❌ FAIL: Invalid move was allowed!")
        return False
    except ValueError as e:
        if "out of bounds" in str(e).lower():
            print(f"✅ PASS: Invalid move correctly rejected: {e}")
            return True
        else:
            print(f"❌ FAIL: Wrong error raised: {e}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wall_valid_move():
    """Test that valid Wall moves still work."""
    print("\nTesting Valid Wall Move...")
    
    board = Board.startpos()
    game = GameState(board=board)
    
    # Clear the board
    for color in [Color.WHITE, Color.BLACK]:
        all_coords = game.cache_manager.occupancy_cache.get_positions(color)
        for coord in all_coords:
            game.cache_manager.occupancy_cache.set_position(coord, None)
    
    # Set game to White's turn
    game.color = Color.WHITE
    
    # Place Wall at [3, 3, 5] (Valid)
    wall_pos = np.array([3, 3, 5])
    game.cache_manager.occupancy_cache.set_position(
        wall_pos,
        np.array([PieceType.WALL, Color.WHITE])
    )
    
    # Move Wall to [4, 3, 5] (Valid - creates 4,3,5; 5,3,5; 4,4,5; 5,4,5)
    valid_move = np.array([3, 3, 5, 4, 3, 5], dtype=np.int16)
    
    print(f"Attempting valid Wall move: {wall_pos} -> {valid_move[3:]}")
    
    try:
        new_state = make_move(game, valid_move)
        print("✅ PASS: Valid move executed successfully")
        return True
    except Exception as e:
        print(f"❌ FAIL: Valid move was rejected: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test1 = test_wall_oob_prevention()
    test2 = test_wall_valid_move()
    
    if test1 and test2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
