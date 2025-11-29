
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.cache.manager import get_cache_manager
from game3d.common.shared_types import Color, PieceType, SIZE, COORD_DTYPE, PIECE_TYPE_DTYPE
from game3d.movement.generator import generate_legal_moves

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_board(cache_manager):
    """Helper to clear the board."""
    cache_manager.occupancy_cache.clear()
    # Reset priest counts
    cache_manager.occupancy_cache._priest_count.fill(0)

def test_stalemate_with_many_pieces():
    print("Testing stalemate with many pieces (0 Priests)...")
    
    # 1. Setup Board
    board = Board()
    cache_manager = get_cache_manager(board)
    clear_board(cache_manager) # Ensure empty
    game_state = GameState(board, Color.WHITE, cache_manager)
    
    # 2. Place pieces
    # Place White King
    king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(king_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
    
    # Place many other pieces (Pawns, Rooks, etc.)
    # Fill up a 4x4x4 block (64 squares) - 1 (King) = 63 pieces
    count = 1
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                if count >= 63:
                    break
                
                pos = np.array([x, y, z], dtype=COORD_DTYPE)
                if np.array_equal(pos, king_pos):
                    continue
                
                # Place a Rook (slider)
                cache_manager.occupancy_cache.set_position(pos, np.array([PieceType.ROOK.value, Color.WHITE.value]))
                count += 1
            if count >= 63: break
        if count >= 63: break
        
    print(f"Placed {count} pieces.")
    print(f"Priest count: {cache_manager.occupancy_cache.get_priest_count(Color.WHITE)}")
    
    # 3. Generate Moves
    moves = generate_legal_moves(game_state)
    print(f"Generated {len(moves)} moves.")
    
    if len(moves) == 0:
        print("FAILURE: Generated 0 moves with 63 pieces!")
    else:
        print("SUCCESS: Generated moves.")

def test_checkmate_logging():
    print("\nTesting Checkmate logging...")
    
    # 1. Setup Board
    board = Board()
    cache_manager = get_cache_manager(board)
    clear_board(cache_manager)
    game_state = GameState(board, Color.WHITE, cache_manager)
    
    # 2. Place pieces for Checkmate
    # White King at (0, 0, 0)
    king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(king_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
    
    # Black Rooks surrounding King to ensure checkmate
    # Attack from (5, 0, 0)
    cache_manager.occupancy_cache.set_position(np.array([5, 0, 0], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    # Block escape to (1, 0, 0)
    cache_manager.occupancy_cache.set_position(np.array([5, 1, 0], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    # Block escape to (0, 1, 0)
    cache_manager.occupancy_cache.set_position(np.array([1, 5, 0], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    # Block escape to (0, 0, 1)
    cache_manager.occupancy_cache.set_position(np.array([0, 0, 5], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    
    # Add diagonal blockers just in case
    cache_manager.occupancy_cache.set_position(np.array([1, 1, 0], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    cache_manager.occupancy_cache.set_position(np.array([1, 0, 1], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    cache_manager.occupancy_cache.set_position(np.array([0, 1, 1], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    cache_manager.occupancy_cache.set_position(np.array([1, 1, 1], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))

    print(f"Priest count: {cache_manager.occupancy_cache.get_priest_count(Color.WHITE)}")
    
    # 3. Generate Moves
    moves = generate_legal_moves(game_state)
    print(f"Generated {len(moves)} moves.")
    
    if len(moves) == 0:
        print("SUCCESS: 0 moves generated (Checkmate). Check logs for 'Checkmate detected'.")
    else:
        print(f"FAILURE: Generated {len(moves)} moves! Not checkmate.")
        # for move in moves: print(move)

if __name__ == "__main__":
    test_stalemate_with_many_pieces()
    test_checkmate_logging()
