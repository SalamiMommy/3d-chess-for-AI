#!/usr/bin/env python3
"""
Reproduction script for "Attackers: Unknown" error.
Scenario:
- White King at (3,2,2)
- Black King at (3,3,3)
- Priests=0 for both
- Expected: is_check(White) == True (because Black King attacks)
- Expected: get_attackers(White) returns ["KING at (3,3,3)"]
- Current Bug: get_attackers(White) returns []
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.attacks.check import king_in_check, get_attackers
from game3d.game.terminal import is_check

logging.basicConfig(level=logging.INFO)

def test_unknown_attacker():
    print("=== Testing Unknown Attacker Bug ===")
    
    # 1. Setup Board
    board = Board.startpos()
    game = GameState(board, Color.WHITE)
    occ_cache = game.cache_manager.occupancy_cache
    
    # Clear board
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    for c in coords:
        occ_cache.set_position(c, None)
        
    # Set positions
    w_king_pos = np.array([3, 2, 2], dtype=COORD_DTYPE)
    b_king_pos = np.array([3, 3, 3], dtype=COORD_DTYPE)
    
    occ_cache.set_position(w_king_pos, np.array([PieceType.KING, Color.WHITE]))
    occ_cache.set_position(b_king_pos, np.array([PieceType.KING, Color.BLACK]))
    
    # Ensure invalidating cache
    game.cache_manager.move_cache.invalidate()
    
    # 2. Verify State
    print(f"White King: {w_king_pos}")
    print(f"Black King: {b_king_pos}")
    
    # 3. Check is_check
    check_status = is_check(game)
    print(f"is_check(White): {check_status}")

    if not check_status:
        print("FAIL: is_check should be True! (Adjacent Kings should mean White is in check)")
        sys.exit(1)

    # 4. Check get_attackers
    attackers = get_attackers(game)
    print(f"Attackers: {attackers}")
        
    return True

if __name__ == "__main__":
    if test_unknown_attacker():
        sys.exit(0)
    else:
        sys.exit(1)
