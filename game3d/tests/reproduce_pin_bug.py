
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.movement.generator import generate_legal_moves
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.attacks.check import king_in_check, batch_moves_leave_king_in_check_fused

def reproduce():
    board = Board.startpos()
    state = GameState(board, Color.WHITE)
    occ = state.cache_manager.occupancy_cache
    
    # Clear board
    coords, _, _ = occ.get_all_occupied_vectorized()
    occ.batch_set_positions(coords, np.zeros((len(coords), 2), dtype=np.int32))
    
    # Setup Pin Scenario:
    # White King at (0,0,0)
    # White Pawn at (0,0,2) (Blocking)
    # Black Rook at (0,0,5) (Attacking King through Pawn)
    
    occ.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
    occ.set_position(np.array([0,0,2]), np.array([PieceType.PAWN, Color.WHITE]))
    occ.set_position(np.array([0,0,5]), np.array([PieceType.ROOK, Color.BLACK]))
    
    # Rebuild state logic
    state._zkey = state.cache_manager._compute_initial_zobrist(state.color)
    state.cache_manager.move_cache.invalidate()
    
    # Verify King is NOT in check (Pawn blocks)
    in_check = king_in_check(state.board, state.color, state.color, state.cache_manager)
    print(f"Is White in check? {in_check}")
    
    if in_check:
        print("Setup failed: King should not be in check (Pawn blocks)")
        return False
        
    print("Generating moves for White...")
    
    # We expect Pawn moves to be ILLEGAL (pinned)
    # Pawn at (0,0,2) usually moves to (0,0,3) or captures diagonally.
    # Moving to (0,0,3) is actually SAFE (still blocks).
    # But capturing or moving sideways would be ILLEGAL.
    # Let's verify what moves are generated.
    # Since it's a Pawn, it might only move forward.
    # Let's use a Rook instead of Pawn to have more freedom to move SIDEWAYS.
    
    occ.set_position(np.array([0,0,2]), np.array([PieceType.ROOK, Color.WHITE]))
    # White Rook at (0,0,2). Can move to (1,0,2) etc.
    # Moving to (1,0,2) unblocks the file.
    
    moves = generate_legal_moves(state)
    print(f"Generated {len(moves)} moves")
    
    # Check for illegal move: moving Rook to (1,0,2)
    # Rook is at (0,0,2)
    found_illegal = False
    
    for m in moves:
        fx, fy, fz, tx, ty, tz = m
        if fx == 0 and fy == 0 and fz == 2:
            # Moving from (0,0,2)
            if tx != 0 or ty != 0:
                # Moving off the z-axis (0,0,x) unblocks the pin!
                print(f"FOUND ILLEGAL PINNED MOVE: {m}")
                found_illegal = True
    
    if found_illegal:
        print("BUG REPRODUCED: Pinned piece allowed to move off-line")
        return True
    else:
        print("Bug not reproduced (Pinned moves filtered correctly)")
        return False

if __name__ == "__main__":
    if reproduce():
        exit(0)
    else:
        exit(1)
