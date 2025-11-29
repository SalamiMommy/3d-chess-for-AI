
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.cache.manager import get_cache_manager
from game3d.common.shared_types import Color, PieceType, SIZE, COORD_DTYPE, PIECE_TYPE_DTYPE
from game3d.movement.generator import generate_legal_moves

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def setup_board_with_many_pieces(cache_manager, color):
    # Place King
    king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(king_pos, np.array([PieceType.KING.value, color.value]))
    
    # Place many pieces
    count = 1
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                if count >= 100: break
                pos = np.array([x, y, z], dtype=COORD_DTYPE)
                if np.array_equal(pos, king_pos): continue
                
                cache_manager.occupancy_cache.set_position(pos, np.array([PieceType.ROOK.value, color.value]))
                count += 1
            if count >= 100: break
        if count >= 100: break
    return count

def test_missing_king():
    print("\n--- Test: Missing King ---")
    board = Board()
    cache_manager = get_cache_manager(board)
    game_state = GameState(board, Color.WHITE, cache_manager)
    
    setup_board_with_many_pieces(cache_manager, Color.WHITE)
    
    # Manually remove King from cache (simulate corruption)
    # We set the square to Empty, so the King is gone.
    king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(king_pos, None)
    
    # Ensure Priest count is 0 (so we check for King)
    # (Setup didn't add priests, so count is 0)
    
    print(f"Priest count: {cache_manager.occupancy_cache.get_priest_count(Color.WHITE)}")
    
    moves = generate_legal_moves(game_state)
    print(f"Moves generated with Missing King: {len(moves)}")
    
    if len(moves) == 0:
        print("SUCCESS: Missing King causes 0 moves.")
    else:
        print(f"FAILURE: Generated {len(moves)} moves despite missing king.")

def test_phantom_check_no_priests():
    print("\n--- Test: Phantom Check (No Priests) ---")
    board = Board()
    cache_manager = get_cache_manager(board)
    game_state = GameState(board, Color.WHITE, cache_manager)
    
    setup_board_with_many_pieces(cache_manager, Color.WHITE)
    
    # No priests.
    # We need to simulate a state where the system THINKS the king is in check,
    # but no move can resolve it.
    
    # Place King at 0,0,0
    # Place Enemy Rooks at 1,0,0 and 0,1,0 and 0,0,1
    cache_manager.occupancy_cache.set_position(np.array([1, 0, 0], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    cache_manager.occupancy_cache.set_position(np.array([0, 1, 0], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    cache_manager.occupancy_cache.set_position(np.array([0, 0, 1], dtype=COORD_DTYPE), np.array([PieceType.ROOK.value, Color.BLACK.value]))
    
    moves = generate_legal_moves(game_state)
    print(f"Moves generated with Unresolvable Check: {len(moves)}")
    
    if len(moves) == 0:
        print("SUCCESS: Unresolvable Check causes 0 moves (Checkmate).")
    else:
        print(f"FAILURE: Generated {len(moves)} moves.")

def test_frozen_pieces():
    print("\n--- Test: Frozen Pieces ---")
    board = Board()
    cache_manager = get_cache_manager(board)
    game_state = GameState(board, Color.WHITE, cache_manager)
    
    setup_board_with_many_pieces(cache_manager, Color.WHITE)
    
    # Add Priests so we ignore check
    priest_pos = np.array([5, 5, 5], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(priest_pos, np.array([PieceType.PRIEST.value, Color.WHITE.value]))
    
    # Freeze White
    # We need to place a BLACK Freezer that affects White pieces.
    # Place Black Freezer at 2,2,2 (inside the block of White pieces)
    freezer_pos = np.array([2, 2, 2], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(freezer_pos, np.array([PieceType.FREEZER.value, Color.BLACK.value]))
    
    # Trigger freeze for BLACK (active player)
    cache_manager.consolidated_aura_cache.trigger_freeze(Color.BLACK, game_state.turn_number)
    
    # Verify freeze
    # White pieces within radius 2 of 2,2,2 should be frozen.
    # Since we packed them tightly, many should be frozen.
    # But maybe not ALL 100 pieces?
    # To freeze ALL, we need multiple freezers or a super freezer.
    # Let's just check if move count DECREASES significantly.
    
    moves = generate_legal_moves(game_state)
    print(f"Moves generated with Frozen Pieces: {len(moves)}")
    
    if len(moves) < 900: # We expect significant reduction from ~990
        print("SUCCESS: Frozen Pieces reduced move count (Freeze is working).")
        if len(moves) == 0:
             print("SUCCESS: Frozen Pieces caused 0 moves!")
    else:
        print(f"FAILURE: Generated {len(moves)} moves. (Freeze might not be working)")

if __name__ == "__main__":
    test_missing_king()
    test_phantom_check_no_priests()
    test_frozen_pieces()
