#!/usr/bin/env python3
"""Minimal debug: trace exact failure point for Black King moves."""

import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.common.coord_utils import coords_to_keys

def debug():
    print("\n=== Final Debug: Black King Move Generation ===\n")
    
    # Setup
    board = Board.startpos()
    game = GameState(board, Color.WHITE)
    occ_cache = game.cache_manager.occupancy_cache
    move_cache = game.cache_manager.move_cache
    
    # Clear board
    print("Step 1: Clearing board...")
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    for c in coords:
        occ_cache.set_position(c, None)
    
    # Verify empty
    count = len(list(occ_cache.get_all_occupied_vectorized()[0]))
    print(f"  Board empty: {count == 0} (count={count})")
    
    # Check positions indices after clear
    print(f"  White positions set size: {len(occ_cache._positions_indices[0])}")
    print(f"  Black positions set size: {len(occ_cache._positions_indices[1])}")
    
    # Place pieces
    print("\nStep 2: Placing pieces...")
    w_king = np.array([1, 1, 1], dtype=COORD_DTYPE)
    b_king = np.array([2, 2, 1], dtype=COORD_DTYPE)
    
    occ_cache.set_position(w_king, np.array([PieceType.KING, Color.WHITE]))
    print(f"  Placed White King at {w_king}")
    print(f"  White positions set size: {len(occ_cache._positions_indices[0])}")
    
    occ_cache.set_position(b_king, np.array([PieceType.KING, Color.BLACK]))
    print(f"  Placed Black King at {b_king}")
    print(f"  Black positions set size: {len(occ_cache._positions_indices[1])}")
    
    # Check get_positions
    print("\nStep 3: Testing get_positions...")
    white_positions = occ_cache.get_positions(Color.WHITE)
    black_positions = occ_cache.get_positions(Color.BLACK)
    print(f"  get_positions(WHITE) returns: {white_positions} (len={len(white_positions)})")
    print(f"  get_positions(BLACK) returns: {black_positions} (len={len(black_positions)})")
    
    # Invalidate cache
    print("\nStep 4: Invalidating cache...")
    move_cache.invalidate()
    
    # Try generating moves for Black via generator
    print("\nStep 5: Generating Black's pseudolegal moves...")
    from game3d.movement.generator import LegalMoveGenerator
    generator = LegalMoveGenerator()
    
    game.color = Color.BLACK
    
    # Check what _generate_raw_pseudolegal returns
    print("  Calling _generate_raw_pseudolegal...")
    raw_moves = generator._generate_raw_pseudolegal(game)
    print(f"  Raw pseudolegal moves: {len(raw_moves)}")
    
    if len(raw_moves) > 0:
        print(f"  First 5: {raw_moves[:5]}")
    else:
        print("  ISSUE: 0 raw pseudolegal moves for Black!")
        
        # Debug deeper
        print("\n  Debugging _generate_raw_pseudolegal internals...")
        all_coords = occ_cache.get_positions(game.color)
        print(f"    all_coords from get_positions: {all_coords}")
        
        if all_coords.size == 0:
            print("    ISSUE: get_positions returned empty!")
        else:
            all_keys = coords_to_keys(all_coords)
            print(f"    all_keys: {all_keys}")
            
            clean_moves, dirty_indices = move_cache.get_incremental_state(game.color, all_keys)
            print(f"    clean_moves count: {len(clean_moves)}")
            print(f"    dirty_indices: {dirty_indices}")
    
    # Compare with White
    print("\nStep 6: Compare with White's moves...")
    game.color = Color.WHITE
    move_cache.invalidate()
    
    white_raw = generator._generate_raw_pseudolegal(game)
    print(f"  White raw pseudolegal moves: {len(white_raw)}")

if __name__ == "__main__":
    debug()
