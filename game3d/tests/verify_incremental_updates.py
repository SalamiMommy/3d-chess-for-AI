
import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.game3d import OptimizedGame3D
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.game.turnmove import make_move
from game3d.movement.generator import generate_legal_moves

def verify_incremental_updates():
    print("Initializing Game...")
    board = Board()
    cache_manager = OptimizedCacheManager(board)
    game = OptimizedGame3D(board=board, cache=cache_manager)
    state = game.state
    
    # 1. Initial Generation
    print("\n--- 1. Initial Generation ---")
    moves = generate_legal_moves(state)
    print(f"Generated {len(moves)} moves.")
    
    # Verify cache state
    affected = cache_manager.move_cache.get_affected_pieces(state.color)
    print(f"Affected pieces after generation: {affected.size} (Expected: 0)")
    if affected.size != 0:
        print("FAILURE: Affected pieces should be 0 after generation")
        return

    # 2. Make a Move (White Pawn Move)
    # Find a pawn move
    pawn_moves = [m for m in moves if state.cache_manager.occupancy_cache.get_type_at(*m[:3]) == PieceType.PAWN]
    if not pawn_moves:
        print("FAILURE: No pawn moves found")
        return
        
    move = pawn_moves[0]
    print(f"\n--- 2. Making Move: {move[:3]} -> {move[3:]} ---")
    
    # Execute move
    make_move(state, move)
    
    # Verify affected pieces
    # After make_move, we expect affected pieces for BOTH colors (current and opponent)
    # Note: make_move switches turn, so state.color is now Black
    
    current_color = state.color # Black
    opp_color = Color.WHITE if current_color == Color.BLACK else Color.BLACK # White
    
    affected_current = cache_manager.move_cache.get_affected_pieces(current_color)
    affected_opp = cache_manager.move_cache.get_affected_pieces(opp_color)
    
    print(f"Current Color ({current_color.name}) Affected: {affected_current.size}")
    print(f"Opponent Color ({opp_color.name}) Affected: {affected_opp.size}")
    
    if affected_current.size == 0 and affected_opp.size == 0:
        print("FAILURE: No pieces marked as affected after move!")
        # Debug: check if manual deletion happened
        # We can't easily check deleted keys without knowing what was there, 
        # but affected_pieces should track dependencies.
        return

    # 3. Regenerate Moves (Black's turn)
    print(f"\n--- 3. Regenerating Moves for {current_color.name} ---")
    moves_black = generate_legal_moves(state)
    print(f"Generated {len(moves_black)} moves for Black.")
    
    # Verify affected pieces cleared for Black
    affected_current_after = cache_manager.move_cache.get_affected_pieces(current_color)
    print(f"Affected pieces for {current_color.name} after generation: {affected_current_after.size} (Expected: 0)")
    
    if affected_current_after.size != 0:
        print("FAILURE: Affected pieces not cleared after generation")
        return

    # 4. Verify Opponent (White) Cache Status
    # White's cache should still be "dirty" (affected pieces present) until we generate for White
    # OR if we switch back and generate.
    # Actually, generate_legal_moves only generates for state.color.
    
    affected_opp_after = cache_manager.move_cache.get_affected_pieces(opp_color)
    print(f"Affected pieces for {opp_color.name} (inactive): {affected_opp_after.size}")
    
    if affected_opp_after.size == 0:
        print("WARNING: Opponent affected pieces cleared? (Maybe okay if no dependencies, but unlikely)")
        
    print("\nSUCCESS: Incremental update flow verified!")

if __name__ == "__main__":
    verify_incremental_updates()
