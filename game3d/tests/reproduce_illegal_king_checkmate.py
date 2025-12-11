
import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.movement.generator import generate_legal_moves

logging.basicConfig(level=logging.INFO)

def reproduce():
    print("Initializing Board...")
    
    # Create scratch board
    board = Board.startpos()
    
    # Create game state
    game = GameState(board, Color.WHITE)
    occ_cache = game.cache_manager.occupancy_cache
    
    # Clear board
    print("Clearing board...")
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    coords_list = [coords[i] for i in range(len(coords))]
    for c in coords_list:
        occ_cache.set_position(c, None)
        
    # Verify empty
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    assert len(coords) == 0, "Board not empty"
    
    # Ensure no priests
    assert occ_cache.get_priest_count(Color.WHITE) == 0
    assert occ_cache.get_priest_count(Color.BLACK) == 0

    print("Setting up scenario...")
    # Black King at (5, 4, 4)
    b_king_pos = np.array([5, 4, 4], dtype=COORD_DTYPE)
    occ_cache.set_position(b_king_pos, np.array([PieceType.KING, Color.BLACK]))
    
    # Black Pawn at (5, 3, 4) (Shield)
    b_pawn_pos = np.array([5, 3, 4], dtype=COORD_DTYPE)
    occ_cache.set_position(b_pawn_pos, np.array([PieceType.PAWN, Color.BLACK]))
    
    # White King at (5, 2, 4)
    w_king_pos = np.array([5, 2, 4], dtype=COORD_DTYPE)
    occ_cache.set_position(w_king_pos, np.array([PieceType.KING, Color.WHITE]))
    
    # Update cache initially
    game.cache_manager.move_cache.invalidate()
    
    # Force generate opponent moves first (to simulate normal game flow)
    # This populates the cache with Black's moves, where Black King is BLOCKED by Black Pawn
    print("Generating Black moves (to populate cache)...")
    # Switch to Black temporarily to generate moves
    game.color = Color.BLACK
    b_moves = generate_legal_moves(game)
    game.color = Color.WHITE
    
    # Verify Black King does NOT have a move to (5, 3, 4)
    b_king_moves_to_pawn = False
    for m in b_moves:
        if np.array_equal(m[:3], b_king_pos) and np.array_equal(m[3:], b_pawn_pos):
            b_king_moves_to_pawn = True
            break
            
    if b_king_moves_to_pawn:
        print("INFO: Black King generates move to friendly pawn? This is unexpected for LEGAL moves.")
    else:
        print("INFO: Black King does not generate legal move to friendly pawn (Expected).")
        
    # NOW: Generate White moves
    print("Generating White moves...")
    w_moves = generate_legal_moves(game)
    
    # Check if White King can capture Pawn at (5, 3, 4)
    # The move: (5, 2, 4) -> (5, 3, 4)
    can_capture = False
    for m in w_moves:
        if np.array_equal(m[:3], w_king_pos) and np.array_equal(m[3:], b_pawn_pos):
            can_capture = True
            break
            
    print(f"Can White King capture Pawn? {can_capture}")
    
    if can_capture:
        print("FAILURE: White King allowed to capture Pawn, moving adjacent to Black King!")
        print("This implies the check detection failed to see the Black King attacks (5, 3, 4).")
        return True # Reproduced
        
    print("SUCCESS: White King disallowed from capturing Pawn (Check correctly detected).")
    return False

if __name__ == "__main__":
    reproduced = reproduce()
    if reproduced:
        print("Issue Reproduced.")
        exit(0)
    else:
        print("Issue NOT Reproduced.")
        exit(1)
