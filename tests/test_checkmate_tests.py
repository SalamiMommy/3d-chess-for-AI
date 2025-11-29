
import pytest
import numpy as np
import sys
import os

# Add current directory to path so we can import game3d
sys.path.append(os.getcwd())

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, Result, COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE
from game3d.cache.manager import OptimizedCacheManager

def force_opponent_move_generation(state):
    """Force generation of opponent moves to populate cache for check detection."""
    opp_color = Color.WHITE if state.color == Color.BLACK else Color.BLACK
    # Create temporary state for opponent
    opp_state = GameState(board=state.board, color=opp_color, cache_manager=state.cache_manager)
    # Access legal_moves to trigger generation and caching
    _ = opp_state.legal_moves

def setup_empty_board():
    """Helper to create a game state with an empty board."""
    board = Board.empty()
    # Create empty initial data
    empty_coords = np.empty((0, 3), dtype=COORD_DTYPE)
    empty_types = np.empty(0, dtype=PIECE_TYPE_DTYPE)
    empty_colors = np.empty(0, dtype=COLOR_DTYPE)
    initial_data = (empty_coords, empty_types, empty_colors)
    
    cache = OptimizedCacheManager(board, Color.WHITE, initial_data=initial_data)
    state = GameState(board=board, color=Color.WHITE, cache_manager=cache)
    return state

def test_checkmate_basic():
    """
    Test basic checkmate scenario:
    - White King at (0,0,0)
    - Black Rook at (0,0,5) (Checks King)
    - Black Rook at (1,0,5) (Guards x=1 column)
    - Black Rook at (0,1,5) (Guards y=1 column)
    - Black Rook at (1,1,5) (Guards (1,1) column)
    """
    state = setup_empty_board()
    cache = state.cache_manager.occupancy_cache
    
    # Setup White King
    cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
    
    # Setup Black Rooks
    cache.set_position(np.array([0,0,5]), np.array([PieceType.ROOK, Color.BLACK]))
    cache.set_position(np.array([1,0,5]), np.array([PieceType.ROOK, Color.BLACK]))
    cache.set_position(np.array([0,1,5]), np.array([PieceType.ROOK, Color.BLACK]))
    cache.set_position(np.array([1,1,5]), np.array([PieceType.ROOK, Color.BLACK]))
    
    # Setup Black King (Required for legal move generation)
    cache.set_position(np.array([8,8,8]), np.array([PieceType.KING, Color.BLACK]))
    
    # Ensure it's White's turn
    state.color = Color.WHITE
    
    # Force opponent (Black) moves to be generated so check detection works
    force_opponent_move_generation(state)
    
    # Verify Game Over
    assert state.is_game_over() == True, "Game should be over (Checkmate)"
    assert state.result() == Result.BLACK_WIN, "Result should be Black Win"

def test_check_but_legal_moves():
    """
    King in check, but has escape.
    White King at (0,0,0).
    Black Rook at (0,0,5).
    King can move to (1,0,0).
    """
    state = setup_empty_board()
    cache = state.cache_manager.occupancy_cache
    
    # Setup White King
    cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
    
    # Setup Black Rook (Check)
    cache.set_position(np.array([0,0,5]), np.array([PieceType.ROOK, Color.BLACK]))
    
    # Setup Black King (Required for legal move generation)
    cache.set_position(np.array([8,8,8]), np.array([PieceType.KING, Color.BLACK]))
    
    # Ensure it's White's turn
    state.color = Color.WHITE
    
    force_opponent_move_generation(state)
    
    # Verify NOT Game Over
    assert state.is_game_over() == False, "Game should NOT be over (King has escape)"

def test_stalemate():
    """
    King NOT in check, but has NO legal moves.
    White King at (0,0,0).
    Black Rooks guarding all escapes but NOT checking.
    """
    state = setup_empty_board()
    cache = state.cache_manager.occupancy_cache
    
    # Setup White King
    cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
    
    # Setup Black Rooks to cage the King
    cache.set_position(np.array([1,0,5]), np.array([PieceType.ROOK, Color.BLACK])) # Guards x=1
    cache.set_position(np.array([0,1,5]), np.array([PieceType.ROOK, Color.BLACK])) # Guards y=1
    cache.set_position(np.array([1,1,5]), np.array([PieceType.ROOK, Color.BLACK])) # Guards (1,1)
    
    cache.set_position(np.array([5,0,1]), np.array([PieceType.ROOK, Color.BLACK])) # Guards z=1, y=0
    cache.set_position(np.array([5,1,1]), np.array([PieceType.ROOK, Color.BLACK])) # Guards z=1, y=1
    
    # Setup Black King (Required for legal move generation)
    cache.set_position(np.array([8,8,8]), np.array([PieceType.KING, Color.BLACK]))
    
    # Ensure it's White's turn
    state.color = Color.WHITE
    
    force_opponent_move_generation(state)
    
    # Verify Stalemate
    # Note: is_game_over() returns True for Stalemate too.
    assert state.is_game_over() == True, "Game should be over (Stalemate)"
    assert state.result() == Result.DRAW, "Result should be Draw (Stalemate)"

def test_friendly_capture_prevents_checkmate():
    """
    King in check, King has NO moves.
    BUT a friendly piece can CAPTURE the attacker.
    
    Setup:
    White King at (0,0,0).
    Black Rook at (0,0,5) (Checking).
    Black Rooks at (0,1,5), (1,1,5) (Guarding escapes).
    
    White Rook at (5,0,5).
    Can capture Black Rook at (0,0,5).
    """
    state = setup_empty_board()
    cache = state.cache_manager.occupancy_cache
    
    # Setup White King
    cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
    
    # Setup Black Rooks (Checkmate pattern)
    cache.set_position(np.array([0,0,5]), np.array([PieceType.ROOK, Color.BLACK])) # Attacker
    # Note: Removed (1,0,5) as it would block the capture from (5,0,5)
    cache.set_position(np.array([0,1,5]), np.array([PieceType.ROOK, Color.BLACK]))
    cache.set_position(np.array([1,1,5]), np.array([PieceType.ROOK, Color.BLACK]))
    
    # Setup Black King (Required for legal move generation)
    cache.set_position(np.array([8,8,8]), np.array([PieceType.KING, Color.BLACK]))
    
    # Setup White Rook (Savior)
    cache.set_position(np.array([5,0,5]), np.array([PieceType.ROOK, Color.WHITE]))
    
    # Ensure it's White's turn
    state.color = Color.WHITE
    
    force_opponent_move_generation(state)
    
    # Verify NOT Game Over
    assert state.is_game_over() == False, "Game should NOT be over (White Rook can capture Attacker)"

def test_friendly_block_prevents_checkmate():
    """
    King in check, King has NO moves.
    BUT a friendly piece can BLOCK the attack.
    
    Setup:
    White King at (0,0,0).
    Black Rook at (0,0,5) (Checking).
    Black Rooks at (1,0,5), (0,1,5), (1,1,5) (Guarding escapes).
    
    White Rook at (5,0,2).
    Can move to (0,0,2) to BLOCK the check from (0,0,5).
    """
    state = setup_empty_board()
    cache = state.cache_manager.occupancy_cache
    
    # Setup White King
    cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
    
    # Setup Black Rooks (Checkmate pattern)
    cache.set_position(np.array([0,0,5]), np.array([PieceType.ROOK, Color.BLACK])) # Attacker
    cache.set_position(np.array([1,0,5]), np.array([PieceType.ROOK, Color.BLACK]))
    cache.set_position(np.array([0,1,5]), np.array([PieceType.ROOK, Color.BLACK]))
    cache.set_position(np.array([1,1,5]), np.array([PieceType.ROOK, Color.BLACK]))
    
    # Setup Black King (Required for legal move generation)
    cache.set_position(np.array([8,8,8]), np.array([PieceType.KING, Color.BLACK]))
    
    # Setup White Rook (Blocker)
    cache.set_position(np.array([5,0,2]), np.array([PieceType.ROOK, Color.WHITE]))
    
    # Ensure it's White's turn
    state.color = Color.WHITE
    
    force_opponent_move_generation(state)
    
    # Verify NOT Game Over
    assert state.is_game_over() == False, "Game should NOT be over (White Rook can block)"

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_checkmate_basic()
        print("test_checkmate_basic PASSED")
        test_check_but_legal_moves()
        print("test_check_but_legal_moves PASSED")
        test_stalemate()
        print("test_stalemate PASSED")
        test_friendly_capture_prevents_checkmate()
        print("test_friendly_capture_prevents_checkmate PASSED")
        test_friendly_block_prevents_checkmate()
        print("test_friendly_block_prevents_checkmate PASSED")
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
