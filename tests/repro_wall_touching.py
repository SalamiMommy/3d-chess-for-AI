#!/usr/bin/env python3
"""
Reproduction script for "Touching Walls" fragmentation bug.
Scenario:
- Place Wall 1 at anchor (0,0,0) -> Occupies (0,0), (1,0), (0,1), (1,1)
- Place Wall 2 at anchor (2,0,0) -> Occupies (2,0), (3,0), (2,1), (3,1)
- Capture piece at (2,0,0) (Anchor of Wall 2)
- Expected: ENTIRE Wall 2 is removed.
- Actual (Bug): Only (2,0,0) is removed, leaving (3,0), (2,1), (3,1) as "ghost" wall pieces.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, PIECE_TYPE_DTYPE
from game3d.core.move_logic import calculate_move_effects
from game3d.core.buffer import state_to_buffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repro")

def test_touching_walls():
    logger.info("=== Testing Touching Walls Fragmentation ===")
    
    # 1. Setup Board
    board = Board.startpos()
    game = GameState(board, Color.WHITE)
    occ_cache = game.cache_manager.occupancy_cache
    
    # Clear board
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    empty_data = np.zeros((len(coords), 2), dtype=PIECE_TYPE_DTYPE)
    occ_cache.batch_set_positions(coords, empty_data)
        
    # 2. Place Adjacent Walls
    # Wall 1 at (0,0,0)
    w1_anchor = np.array([0, 0, 0], dtype=COORD_DTYPE)
    w1_parts = [
        [0,0,0], [1,0,0], [0,1,0], [1,1,0]
    ]
    
    # Wall 2 at (2,0,0) - Adjacent to Wall 1
    w2_anchor = np.array([2, 0, 0], dtype=COORD_DTYPE)
    w2_parts = [
        [2,0,0], [3,0,0], [2,1,0], [3,1,0]
    ]
    
    # Set pieces
    wall_data = np.array([[PieceType.WALL, Color.WHITE]] * 8, dtype=PIECE_TYPE_DTYPE)
    all_wall_coords = np.array(w1_parts + w2_parts, dtype=COORD_DTYPE)
    
    occ_cache.batch_set_positions(all_wall_coords, wall_data)
    
    # Verify setup
    t, _ = occ_cache.get_fast(np.array([1, 0, 0]))
    if t != PieceType.WALL:
        logger.error("Setup failed: Wall 1 not present")
        return False
        
    t, _ = occ_cache.get_fast(np.array([2, 0, 0]))
    if t != PieceType.WALL:
        logger.error("Setup failed: Wall 2 not present")
        return False

    logger.info("Walls placed successfully.")

    # 3. Simulate Capture of Wall 2 Anchor (2,0,0)
    # We use calculate_move_effects directly to see what WOULD happen
    # Move: pawn (placed at 2,0,1) captures (2,0,0)
    
    # Place attacker
    attacker_pos = np.array([2, 0, 1], dtype=COORD_DTYPE)
    occ_cache.set_position(attacker_pos, np.array([PieceType.PAWN, Color.BLACK]))
    
    # Construct move
    move = np.array([2, 0, 1, 2, 0, 0], dtype=COORD_DTYPE)
    
    # Create buffer and calculate effects
    buffer = state_to_buffer(game, readonly=True)
    effects = calculate_move_effects(move, buffer)
    
    # 4. Analyze Effects
    # We expect coords_to_clear (or update type=0) to contain ALL parts of Wall 2
    # except the one being captured (which is overwritten by attacker).
    # Wait, capture logic usually overwrites target, and clears "others".
    
    logger.info(f"Coords to clear: {effects.coords_to_clear}")
    logger.info(f"Coords to update: {effects.coords_to_update}")
    
    # Collect all cleared/overwritten squares
    cleared_set = set()
    for c in effects.coords_to_clear:
        cleared_set.add(tuple(c))
        
    for i, c in enumerate(effects.coords_to_update):
        # type=0 means clear
        if effects.new_pieces_data[i][0] == 0: 
            cleared_set.add(tuple(c))
        # standard update (attacker moves to target) overwrites 2,0,0
        if np.array_equal(c, w2_anchor):
             cleared_set.add(tuple(c))

    # Check if all w2 parts are handled
    missing_parts = []
    for p in w2_parts:
        if tuple(p) not in cleared_set:
            missing_parts.append(p)
            
    if missing_parts:
        logger.error(f"FAILURE: Wall fragmentation! These parts of Wall 2 were NOT removed: {missing_parts}")
        return False
        
    logger.info("SUCCESS: All wall parts correctly marked for removal.")
    return True

if __name__ == "__main__":
    if test_touching_walls():
        sys.exit(0)
    else:
        sys.exit(1)
