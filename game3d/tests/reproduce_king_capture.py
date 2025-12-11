
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.movement.generator import generate_legal_moves
from game3d.common.shared_types import Color, PieceType, SIZE

def reproduce():
    # Setup board
    board = Board.startpos()
    # Create game state
    state = GameState(board, Color.WHITE)
    
    # Access via state
    occ = state.cache_manager.occupancy_cache
    
    # Clear board
    # We need to get all occupied coords first
    coords, _, _ = occ.get_all_occupied_vectorized()
    # Create empty update data
    empty_data = np.zeros((len(coords), 2), dtype=np.int32) # (Type, Color) = (0, 0)
    occ.batch_set_positions(coords, empty_data)
    
    # White Rook at (0,0,0)
    # Black King at (0,0,5)
    # Rook can capture King
    
    # Setup
    occ.set_position(np.array([0,0,0]), np.array([PieceType.ROOK, Color.WHITE]))
    occ.set_position(np.array([0,0,5]), np.array([PieceType.KING, Color.BLACK]))
    
    # Rebuild state to be sure
    state._zkey = state.cache_manager._compute_initial_zobrist(state.color)
    state.cache_manager.move_cache.invalidate()
    
    print("Generating moves for White...")
    moves = generate_legal_moves(state)
    
    print(f"Generated {len(moves)} moves")
    
    capture_king = False
    for m in moves:
        # m is [fx, fy, fz, tx, ty, tz]
        if m[3] == 0 and m[4] == 0 and m[5] == 5:
            print("FOUND MOVE CAPTURING KING:", m)
            capture_king = True
            
    if capture_king:
        print("BUG REPRODUCED: King capture allowed")
        return True
    else:
        print("Bug not reproduced (King capture not found)")
        return False

if __name__ == "__main__":
    if reproduce():
        exit(0)
    else:
        exit(1)
