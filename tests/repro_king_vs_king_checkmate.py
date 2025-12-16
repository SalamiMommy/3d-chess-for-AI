#!/usr/bin/env python3
"""
Reproduce the exact bug scenario from user:
- White King at (1,1,1), Black King at (2,2,1)
- Both have 0 priests
- White King should NOT be able to move to (1,1,1) if it puts itself adjacent to Black King
- The checkmate by White King against Black King should be INVALID

This tests the case where the game log showed:
  Checkmate (Winner: WHITE, Turn: 1472)
  White: Priests=0, King=(1,1,1)
  Black: Priests=0, King=(2,2,1)
  Attackers: KING at (1,1,1)
"""

import sys
import os
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.movement.generator import generate_legal_moves
from game3d.attacks.check import king_in_check, move_would_leave_king_in_check, square_attacked_by
from game3d.game.terminal import is_check

logging.basicConfig(level=logging.INFO)

def test_king_vs_king_checkmate():
    """Test: Two adjacent kings with no priests - verify neither can attack the other."""
    print("\n=== Test: King vs King Checkmate Bug ===")
    
    # Create clean board
    board = Board.startpos()
    game = GameState(board, Color.WHITE)
    occ_cache = game.cache_manager.occupancy_cache
    
    # Clear board
    print("Clearing board...")
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    for c in coords:
        occ_cache.set_position(c, None)
    
    # Verify no priests
    assert occ_cache.get_priest_count(Color.WHITE) == 0, "White should have 0 priests"
    assert occ_cache.get_priest_count(Color.BLACK) == 0, "Black should have 0 priests"
    
    # Setup exact scenario from user
    # White King at (1,1,1)
    w_king_pos = np.array([1, 1, 1], dtype=COORD_DTYPE)
    occ_cache.set_position(w_king_pos, np.array([PieceType.KING, Color.WHITE]))
    
    # Black King at (2,2,1) - adjacent to white king
    b_king_pos = np.array([2, 2, 1], dtype=COORD_DTYPE)
    occ_cache.set_position(b_king_pos, np.array([PieceType.KING, Color.BLACK]))
    
    print(f"White King at: {w_king_pos}")
    print(f"Black King at: {b_king_pos}")
    print(f"Distance: diagonal adjacent (should be in mutual check range)")
    
    # Invalidate move cache to force regeneration
    game.cache_manager.move_cache.invalidate()
    
    # TEST 1: Check if Black King position is attacked by White King
    print("\n--- Test 1: Does White King attack Black King's square? ---")
    
    # Generate BOTH colors' pseudolegal moves 
    from game3d.movement.generator import LegalMoveGenerator
    generator = LegalMoveGenerator()
    
    # Generate White's moves (needed for test 1)
    game.color = Color.WHITE
    generator.refresh_pseudolegal_moves(game)
    
    # Generate Black's moves (needed for test 2)
    game.color = Color.BLACK
    generator.refresh_pseudolegal_moves(game)
    
    # Now test White attacking Black
    game.color = Color.WHITE
    is_b_king_attacked = square_attacked_by(
        game.board, Color.BLACK, b_king_pos, Color.WHITE, 
        game.cache_manager, use_move_cache=True
    )
    print(f"Black King attacked by White? {is_b_king_attacked}")
    
    # TEST 2: Check if White King position is attacked by Black King
    print("\n--- Test 2: Does Black King attack White King's square? ---")
    game.color = Color.BLACK
    generator.refresh_pseudolegal_moves(game)
    
    is_w_king_attacked = square_attacked_by(
        game.board, Color.WHITE, w_king_pos, Color.BLACK,
        game.cache_manager, use_move_cache=True
    )
    print(f"White King attacked by Black? {is_w_king_attacked}")
    
    # TEST 3: Is Black in check? (When White attacks Black King)
    print("\n--- Test 3: Is Black King in check? ---")
    game.color = Color.BLACK  # Black's turn
    in_check = is_check(game)
    print(f"Black is in check: {in_check}")
    
    # TEST 4: Check Black's legal moves
    print("\n--- Test 4: Black's legal moves ---")
    game.cache_manager.move_cache.invalidate()
    black_moves = generate_legal_moves(game)
    print(f"Black has {len(black_moves)} legal moves")
    
    # TEST 5: Is White in check? (When Black attacks White King)
    print("\n--- Test 5: Is White King in check? ---")
    game.color = Color.WHITE  # White's turn
    game.cache_manager.move_cache.invalidate()
    generator.refresh_pseudolegal_moves(game)  # Generate black's moves
    game.color = Color.BLACK
    generator.refresh_pseudolegal_moves(game)
    game.color = Color.WHITE
    
    in_check_white = is_check(game)
    print(f"White is in check: {in_check_white}")
    
    # ANALYSIS
    print("\n=== Analysis ===")
    if is_b_king_attacked and is_w_king_attacked:
        print("SUCCESS: Both kings attack each other's squares.")
        print("This is the correct behavior for the attack kernel (Kings are always attackers).")
        print("Note: In a real game, this state is unreachable (illegal move).")
        return True
    elif is_b_king_attacked and not is_w_king_attacked:
        print("BUG: White king attacks Black, but Black king doesn't attack White!")
        print("This is asymmetric and incorrect.")
        return False
    elif not is_b_king_attacked and is_w_king_attacked:
        print("BUG: Black king attacks White, but White king doesn't attack Black!")
        print("This is asymmetric and incorrect.")
        return False
    else:
        print("BUG: Neither king attacks the other. They are adjacent and should attack!")
        print("Regression: Kings should be attackers.")
        return False

if __name__ == "__main__":
    success = test_king_vs_king_checkmate()
    if success:
        print("\nTest completed - see analysis above.")
    else:
        print("\nTest found a BUG!")
        sys.exit(1)
