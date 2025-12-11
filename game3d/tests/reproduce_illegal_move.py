
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.movement.generator import generate_legal_moves
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.attacks.check import king_in_check

def reproduce():
    board = Board.startpos()
    state = GameState(board, Color.WHITE)
    occ = state.cache_manager.occupancy_cache
    
    # Clear board
    coords, _, _ = occ.get_all_occupied_vectorized()
    occ.batch_set_positions(coords, np.zeros((len(coords), 2), dtype=np.int32))
    
    # Setup Check Scenario:
    # White King at (0,0,0)
    # Black Rook at (0,0,5) -> Attacks (0,0,0)
    
    # White needs to move.
    
    # Valid moves: King moves to (1,0,0), (0,1,0), (1,1,0) etc. (assuming not attacked)
    # Invalid moves: King stays at (0,0,0) (impossible if it's a move), 
    #                BUT if we have another piece, say White Pawn at (5,5,5).
    #                Moving the Pawn is ILLEGAL because King remains in check.
    
    occ.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
    occ.set_position(np.array([0,0,5]), np.array([PieceType.ROOK, Color.BLACK]))
    occ.set_position(np.array([5,5,5]), np.array([PieceType.PAWN, Color.WHITE])) # Irrelevant piece
    
    # Rebuild state logic
    state._zkey = state.cache_manager._compute_initial_zobrist(state.color)
    state.cache_manager.move_cache.invalidate()
    
    # Verify initial check state
    in_check = king_in_check(state.board, state.color, state.color, state.cache_manager)
    print(f"Is White in check? {in_check}")
    
    print("Generating moves for White...")
    moves = generate_legal_moves(state)
    print(f"Generated {len(moves)} moves")
    
    found_illegal = False
    for m in moves:
        # Check if this move is the Pawn move
        if m[0] == 5 and m[1] == 5 and m[2] == 5:
            print(f"FOUND ILLEGAL MOVE: Pawn moved {m}, but King is still in check!")
            found_illegal = True
            
    if found_illegal:
        print("BUG REPRODUCED: Player in check allowed to move non-king piece (staying in check)")
        return True
    else:
        print("Bug not reproduced (Illegal moves filtered correctly)")
        return False

if __name__ == "__main__":
    if reproduce():
        exit(0)
    else:
        exit(1)
