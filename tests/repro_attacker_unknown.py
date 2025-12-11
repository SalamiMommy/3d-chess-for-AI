"""
Reproduction script for the 'Attackers: Unknown' bug.

This script reproduces the scenario where:
1. Black King is at (4,4,8)
2. White has pieces that should be attacking the king
3. 1707 moves are generated but none target the king position

The goal is to identify WHY move destinations don't match the king position.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.common.shared_types import (
    Color, PieceType, COORD_DTYPE, SIZE, MinimalStateProxy
)
from game3d.attacks.check import get_attackers, _find_attackers_of_square
from game3d.core.buffer import state_to_buffer, GameBuffer
from game3d.core.api import generate_pseudolegal_moves


def clear_board(gs: GameState):
    """Clear all pieces from the board using rebuild."""
    occ_cache = gs.cache_manager.occupancy_cache
    occ_cache.rebuild(
        np.empty((0, 3), dtype=np.int16),
        np.empty(0, dtype=np.int8),
        np.empty(0, dtype=np.int8)
    )
    gs.cache_manager.move_cache.invalidate()



def place_piece(gs: GameState, coord: tuple, piece_type: PieceType, color: Color):
    """Place a piece on the board."""
    pos = np.array(coord, dtype=np.int16)
    data = np.array([piece_type.value, int(color)], dtype=np.int8)
    gs.cache_manager.occupancy_cache.set_position(pos, data)


def test_case_1():
    """Test: Simple rook attacking king scenario."""
    print("\n=== TEST CASE 1: Simple Rook Attack ===")
    
    gs = GameState(Board.empty(), Color.BLACK)  # Black's turn (being checked)
    
    # Place pieces
    # White King at (4,4,0)
    place_piece(gs, (4, 4, 0), PieceType.KING, Color.WHITE)
    # White Priest at (2,2,0) - so white has priest protection
    place_piece(gs, (2, 2, 0), PieceType.PRIEST, Color.WHITE)
    # White Rook at (4,4,4) - attacking Black King along z-axis
    place_piece(gs, (4, 4, 4), PieceType.ROOK, Color.WHITE)
    
    # Black King at (4,4,8) - no priests
    place_piece(gs, (4, 4, 8), PieceType.KING, Color.BLACK)
    
    # Refresh caches
    gs.cache_manager.move_cache.invalidate()
    
    print(f"Black King position: (4, 4, 8)")
    print(f"White Rook position: (4, 4, 4)")
    
    # Verify the board state
    occ_cache = gs.cache_manager.occupancy_cache
    print(f"Piece at (4,4,8): {occ_cache.get(np.array([4,4,8]))}")
    print(f"Piece at (4,4,4): {occ_cache.get(np.array([4,4,4]))}")
    
    # Now simulate what get_attackers does
    king_color = gs.color  # Black
    opponent_color = Color(king_color).opposite().value  # White
    
    print(f"\nKing color (Black): {king_color}")
    print(f"Opponent color (White): {opponent_color}")
    
    # Find king position
    king_pos = occ_cache.find_king(king_color)
    print(f"King position found: {king_pos}")
    
    if king_pos is None:
        print("ERROR: King not found!")
        return False
    
    king_pos_arr = king_pos.astype(COORD_DTYPE)
    
    # Create MinimalStateProxy for opponent
    dummy_state = MinimalStateProxy(gs.board, opponent_color, gs.cache_manager)
    
    # Generate buffer
    buffer = state_to_buffer(dummy_state, readonly=True)
    
    print(f"\nBuffer active color (meta[0]): {buffer.meta[0]} (should be {opponent_color} = White)")
    print(f"Buffer occupied_count: {buffer.occupied_count}")
    
    # Show sparse arrays
    print("\nSparse arrays (first N pieces):")
    for i in range(buffer.occupied_count):
        coord = buffer.occupied_coords[i]
        ptype = buffer.occupied_types[i]
        pcolor = buffer.occupied_colors[i]
        print(f"  [{i}] coord={coord}, type={ptype} ({PieceType(ptype).name if ptype > 0 else 'EMPTY'}), color={pcolor}")
    
    # Generate moves
    opponent_moves = generate_pseudolegal_moves(buffer)
    print(f"\nGenerated {opponent_moves.shape[0]} moves")
    
    # Check for moves targeting king position
    kx, ky, kz = king_pos_arr
    print(f"Looking for moves targeting king at ({kx}, {ky}, {kz})...")
    
    attacking_moves = []
    for i in range(opponent_moves.shape[0]):
        move = opponent_moves[i]
        if move[3] == kx and move[4] == ky and move[5] == kz:
            attacking_moves.append(move)
            
    print(f"Found {len(attacking_moves)} moves targeting king")
    
    if attacking_moves:
        print("Attacking moves:")
        for m in attacking_moves:
            print(f"  From ({m[0]},{m[1]},{m[2]}) -> To ({m[3]},{m[4]},{m[5]})")
        return True
    else:
        # Debug: show all move destinations
        print("\n--- DEBUG: All move destinations ---")
        dest_set = set()
        for i in range(min(100, opponent_moves.shape[0])):  # First 100 for brevity
            m = opponent_moves[i]
            dest = (int(m[3]), int(m[4]), int(m[5]))
            dest_set.add(dest)
        
        print(f"Unique destinations (first 100 moves): {len(dest_set)}")
        
        # Check if z=8 appears in any destination
        z8_dests = [d for d in dest_set if d[2] == 8]
        print(f"Destinations with z=8: {z8_dests}")
        
        # Check for destinations with x=4, y=4
        xy44_dests = [d for d in dest_set if d[0] == 4 and d[1] == 4]
        print(f"Destinations with x=4, y=4: {xy44_dests}")
        
        # Check rook moves specifically
        print("\n--- DEBUG: Rook moves from (4,4,4) ---")
        for i in range(opponent_moves.shape[0]):
            m = opponent_moves[i]
            if m[0] == 4 and m[1] == 4 and m[2] == 4:
                print(f"  -> ({m[3]},{m[4]},{m[5]})")
                
        return False


def test_case_2():
    """Test: Complex checkmate scenario similar to actual game."""
    print("\n=== TEST CASE 2: Complex Checkmate Scenario ===")
    
    # The actual bug scenario:
    # Turn 243, Black to move (being checked)
    # White: King at (4,4,0), 1 Priest
    # Black: King at (4,4,8), 0 Priests
    # Checkmate declared but "Attackers: Unknown"
    
    gs = GameState(Board.empty(), Color.BLACK)  # Black's turn
    
    # Setup positions from log
    place_piece(gs, (4, 4, 0), PieceType.KING, Color.WHITE)
    place_piece(gs, (4, 4, 8), PieceType.KING, Color.BLACK)
    # White has 1 priest - place it somewhere
    place_piece(gs, (2, 2, 0), PieceType.PRIEST, Color.WHITE)
    
    # We need an attacker - but we don't know which piece from the logs
    # Try a Queen on same file
    place_piece(gs, (4, 4, 1), PieceType.QUEEN, Color.WHITE)
    
    gs.cache_manager.move_cache.invalidate()
    
    # Get attackers
    attackers = get_attackers(gs, gs.cache_manager)
    print(f"Attackers found: {attackers}")
    
    if not attackers:
        print("BUG REPRODUCED: No attackers found despite checkmate!")
        return True  # Bug reproduced
    else:
        print("Attackers correctly identified")
        return False


def test_case_3():
    """Test: Debug sparse array population."""
    print("\n=== TEST CASE 3: Debug Sparse Array Population ===")
    
    gs = GameState(Board.empty(), Color.BLACK)
    
    # Simple setup
    place_piece(gs, (4, 4, 0), PieceType.KING, Color.WHITE)
    place_piece(gs, (4, 4, 8), PieceType.KING, Color.BLACK)
    place_piece(gs, (4, 4, 4), PieceType.ROOK, Color.WHITE)
    
    gs.cache_manager.move_cache.invalidate()
    
    # Create buffer for WHITE moves (opponent of Black king)
    dummy_state = MinimalStateProxy(gs.board, Color.WHITE, gs.cache_manager)
    buffer = state_to_buffer(dummy_state, readonly=True)
    
    print(f"Buffer meta[0] (active color): {buffer.meta[0]} (1=White, 2=Black)")
    print(f"Buffer occupied_count: {buffer.occupied_count}")
    
    # Check if White pieces are in the sparse array
    white_pieces = 0
    rook_found = False
    rook_idx = -1
    
    for i in range(buffer.occupied_count):
        color = buffer.occupied_colors[i]
        if color == Color.WHITE:
            white_pieces += 1
            coord = buffer.occupied_coords[i]
            ptype = buffer.occupied_types[i]
            print(f"  White piece [{i}]: coord={coord}, type={PieceType(ptype).name}")
            if ptype == PieceType.ROOK.value:
                rook_found = True
                rook_idx = i
    
    print(f"\nTotal White pieces in buffer: {white_pieces}")
    print(f"Rook found at index: {rook_idx}")
    
    if not rook_found:
        print("ERROR: White Rook not in sparse array!")
        return False
    
    # Generate moves and check if rook can reach (4,4,8)
    opponent_moves = generate_pseudolegal_moves(buffer)
    print(f"\nGenerated {opponent_moves.shape[0]} moves")
    
    # Find rook moves
    rook_moves = []
    for i in range(opponent_moves.shape[0]):
        m = opponent_moves[i]
        if m[0] == 4 and m[1] == 4 and m[2] == 4:
            rook_moves.append(m)
    
    print(f"Rook moves found: {len(rook_moves)}")
    for m in rook_moves:
        print(f"  ({m[0]},{m[1]},{m[2]}) -> ({m[3]},{m[4]},{m[5]})")
    
    # Check if (4,4,8) is in destinations
    targets_king = any(m[3] == 4 and m[4] == 4 and m[5] == 8 for m in rook_moves)
    print(f"\nRook targets king at (4,4,8): {targets_king}")
    
    return targets_king


if __name__ == "__main__":
    print("=" * 60)
    print("Reproduction Script: Attackers Unknown Bug")
    print("=" * 60)
    
    test1_passed = test_case_1()
    test2_reproduced = test_case_2()
    test3_passed = test_case_3()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Simple Rook Attack): {'PASS' if test1_passed else 'FAIL - No attacking moves found'}")
    print(f"Test 2 (Complex Checkmate): {'BUG REPRODUCED' if test2_reproduced else 'No bug (attackers found)'}")
    print(f"Test 3 (Sparse Array Debug): {'PASS' if test3_passed else 'FAIL - Rook cannot reach king'}")
    
    if not test1_passed or not test3_passed:
        print("\n!!! BUG IDENTIFIED: Move generation or sparse array issue !!!")
        sys.exit(1)
    else:
        print("\nNo bug found in simple scenarios. Issue may be in complex game state.")
        sys.exit(0)
