
import numpy as np
from game3d.common.shared_types import Color, PieceType
from game3d.game.gamestate import GameState
from game3d.pieces.pieces.infiltrator import generate_infiltrator_moves
from game3d.pieces.pieces.pawn import generate_pawn_moves

def test_pawn_and_infiltrator():
    print("Initializing GameState...")
    game = GameState.from_startpos()
    cache_manager = game.cache_manager
    cache_manager.occupancy_cache.clear()
    
    # --- TEST 1: Pawn Movement (Should be Z-axis) ---
    print("\n--- TEST 1: Pawn Movement ---")
    # Place White Pawn at (4, 4, 2) (Standard start rank for White)
    pawn_pos = np.array([4, 4, 2])
    cache_manager.occupancy_cache.set_position_fast(pawn_pos, PieceType.PAWN.value, Color.WHITE.value)
    
    moves = generate_pawn_moves(cache_manager, Color.WHITE.value, pawn_pos)
    
    found_z_push = False
    found_y_push = False
    
    for move in moves:
        tx, ty, tz = move[3], move[4], move[5]
        # print(f"Pawn Move: {tx}, {ty}, {tz}")
        
        if tx == 4 and ty == 4 and tz == 3:
            found_z_push = True
        if tx == 4 and ty == 5 and tz == 2:
            found_y_push = True
            
    if found_z_push:
        print("SUCCESS: Pawn moves along Z-axis (4, 4, 3)")
    else:
        print("FAILURE: Pawn did NOT move along Z-axis")
        
    if found_y_push:
        print("BUG CONFIRMED: Pawn moves along Y-axis (4, 5, 2)")

    # --- TEST 2: Infiltrator Targeting (Should be in front of Pawn) ---
    print("\n--- TEST 2: Infiltrator Targeting ---")
    cache_manager.occupancy_cache.clear()
    
    # Place White Infiltrator at (0, 0, 0)
    inf_pos = np.array([0, 0, 0])
    cache_manager.occupancy_cache.set_position_fast(inf_pos, PieceType.INFILTRATOR.value, Color.WHITE.value)
    
    # Place Black Pawn at (4, 4, 6) (Standard start rank for Black)
    # Black pawns move -Z (to 5), so front is (4, 4, 5)
    black_pawn_pos = np.array([4, 4, 6])
    cache_manager.occupancy_cache.set_position_fast(black_pawn_pos, PieceType.PAWN.value, Color.BLACK.value)
    
    moves = generate_infiltrator_moves(cache_manager, Color.WHITE.value, inf_pos)
    
    found_front = False # (4, 4, 5)
    found_behind = False # (4, 4, 7)
    
    for move in moves:
        tx, ty, tz = move[3], move[4], move[5]
        
        if tx == 4 and ty == 4 and tz == 5:
            found_front = True
        if tx == 4 and ty == 4 and tz == 7:
            found_behind = True
            
    if found_front:
        print("SUCCESS: Infiltrator targets pawn front (4, 4, 5)")
    else:
        print("FAILURE: Infiltrator did NOT target pawn front")
        
    if found_behind:
        print("BUG CONFIRMED: Infiltrator targets pawn back (4, 4, 7)")

if __name__ == "__main__":
    test_pawn_and_infiltrator()
