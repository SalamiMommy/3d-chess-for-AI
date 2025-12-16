
import numpy as np
import sys
import os

# Ensure game3d is in path
sys.path.append(os.getcwd())

from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color, SIZE, COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE
from game3d.attacks.check import move_would_leave_king_in_check, square_attacked_by
from game3d.core.attacks import is_check
from game3d.cache.manager import get_cache_manager

def clear_board(game):
    # Clear occupancy cache directly
    occ = game.cache_manager.occupancy_cache
    occ._occ.fill(0)
    occ._ptype.fill(0)
    # Rebuild internal indices/caches
    occ.rebuild(np.empty((0,3), dtype=COORD_DTYPE), 
                np.empty(0, dtype=PIECE_TYPE_DTYPE), 
                np.empty(0, dtype=COLOR_DTYPE))
    # Clear other caches
    game.cache_manager.move_cache.clear()
    if hasattr(game.cache_manager, 'trailblaze_cache'):
        # Reset counters
        if hasattr(game.cache_manager.trailblaze_cache, '_victim_counters'):
            game.cache_manager.trailblaze_cache._victim_counters.fill(0)

def place_piece(game, x, y, z, ptype, color):
    occ = game.cache_manager.occupancy_cache
    coord = np.array([[x, y, z]], dtype=COORD_DTYPE)
    piece = np.array([[ptype, color]], dtype=np.int16) # [type, color]
    occ.batch_set_positions(coord, piece)

def test_archer_check():
    print("Testing Archer Check (Range 2)...")
    game = GameState.from_startpos()
    clear_board(game)
    
    # Place White King at 2,2,2
    place_piece(game, 2, 2, 2, PieceType.KING, Color.WHITE)
    
    # Place Black Archer at 4,2,2 (Distance 2)
    # 4-2 = 2. dx=2, dy=0, dz=0. DistSq = 4. Should be check.
    place_piece(game, 4, 2, 2, PieceType.ARCHER, Color.BLACK)
    
    # Check directly using square_attacked_by
    # Logic in check.py should catch presence of Archer and trigger geometric check
    is_attacked = square_attacked_by(game.board, Color.WHITE, np.array([2,2,2]), Color.BLACK.value, game.cache_manager)
    
    print(f"King at (2,2,2), Archer at (4,2,2). Attacked? {is_attacked}")
    if is_attacked:
        print("PASS: Archer check detected.")
    else:
        print("FAIL: Archer check NOT detected.")

def test_bomb_check():
    print("\nTesting Bomb Check (Radius 2)...")
    game = GameState.from_startpos()
    clear_board(game)
    
    # Place White King at 2,2,2
    place_piece(game, 2, 2, 2, PieceType.KING, Color.WHITE)
    
    # Place Black Bomb at 3,3,2 (dx=1, dy=1, dz=0) -> Distance Sqrt(2). Inside Radius 2.
    place_piece(game, 3, 3, 2, PieceType.BOMB, Color.BLACK)
    
    is_attacked = square_attacked_by(game.board, Color.WHITE, np.array([2,2,2]), Color.BLACK.value, game.cache_manager)
    
    print(f"King at (2,2,2), Bomb at (3,3,2). Attacked? {is_attacked}")
    if is_attacked:
        print("PASS: Bomb check detected.")
    else:
        print("FAIL: Bomb check NOT detected.")
        
    # Test Bomb at distance 3 (Safe)
    place_piece(game, 3, 3, 2, 0, 0) # Remove
    place_piece(game, 5, 2, 2, PieceType.BOMB, Color.BLACK) # dx=3. Unsafe? No, 3 > 2.
    
    is_attacked_far = square_attacked_by(game.board, Color.WHITE, np.array([2,2,2]), Color.BLACK.value, game.cache_manager)
    print(f"King at (2,2,2), Bomb at (5,2,2). Attacked? {is_attacked_far}")
    if not is_attacked_far:
        print("PASS: Bomb range limit respected.")
    else:
        print("FAIL: Bomb attacks too far.")

def test_trailblazer_check():
    print("\nTesting Trailblazer Counter Check...")
    game = GameState.from_startpos()
    clear_board(game)
    
    # Place White King at 0,0,0
    place_piece(game, 0, 0, 0, PieceType.KING, Color.WHITE)
    
    # Set lethal counter at 1,0,0
    target_sq = np.array([1, 0, 0], dtype=COORD_DTYPE)
    # Use internal API to set counter
    if hasattr(game.cache_manager, 'trailblaze_cache'):
        tb_cache = game.cache_manager.trailblaze_cache
        # set counter to 2
        # tb_cache has _victim_counters flat array usually
        if hasattr(tb_cache, '_victim_counters'):
            idx = 1 + 0*SIZE + 0*SIZE*SIZE # x + y*S + ...
            tb_cache._victim_counters[idx] = 2
            
        ctr = tb_cache.get_counter(target_sq)
        print(f"Counter at (1,0,0) is {ctr}")
        
    # Check if a move to (1,0,0) is unsafe
    move = np.array([0,0,0, 1,0,0], dtype=np.int16)
    
    is_unsafe_2 = move_would_leave_king_in_check(game, move, game.cache_manager)
    print(f"King Move (0,0,0)->(1,0,0) with Counter={ctr}. Unsafe? {is_unsafe_2}")
    if not is_unsafe_2:
        print("PASS: Trailblazer counter=2 is SAFE (Correct).")
    else:
        print("FAIL: Trailblazer counter=2 is UNSAFE (Incorrect).")

    # Increment one more to 3 (Lethal)
    if hasattr(game.cache_manager, 'trailblaze_cache'):
         target_sq = np.array([1, 0, 0], dtype=COORD_DTYPE)
         game.cache_manager.trailblaze_cache.increment_counter(target_sq) # 3
         
    ctr_3 = game.cache_manager.trailblaze_cache.get_counter(target_sq)
    print(f"Counter at (1,0,0) is {ctr_3}")
    
    is_unsafe_3 = move_would_leave_king_in_check(game, move, game.cache_manager)
    print(f"King Move (0,0,0)->(1,0,0) with Counter={ctr_3}. Unsafe? {is_unsafe_3}")
    if is_unsafe_3:
        print("PASS: Trailblazer counter=3 is UNSAFE (Correct).")
    else:
        print("FAIL: Trailblazer counter=3 is SAFE (Incorrect).")

    # Test standing on lethal counter (King at 2,0,0 with counter 3)
    # Place King at 2,0,0
    place_piece(game, 2, 0, 0, PieceType.KING, Color.WHITE)
    target_sq2 = np.array([2, 0, 0], dtype=COORD_DTYPE)
    tb_cache._victim_counters[2] = 3 # At 2,0,0 (Lethal)
    
    is_at_3 = square_attacked_by(game.board, Color.WHITE, np.array([2,0,0]), Color.BLACK.value, game.cache_manager)
    print(f"King Standing on Counter=3 at (2,0,0). Attacked? {is_at_3}")
    if is_at_3:
        pass
    else:
         print("FAIL: Standing on Counter=3 not detected.")

    tb_cache._victim_counters[2] = 2 # At 2,0,0 (Safe)
    is_at_2 = square_attacked_by(game.board, Color.WHITE, np.array([2,0,0]), Color.BLACK.value, game.cache_manager)
    print(f"King Standing on Counter=2 at (2,0,0). Attacked? {is_at_2}")
    if not is_at_2:
        print("PASS: Standing on Trailblazer tests passed.")
    else:
        print("FAIL: Standing on Counter=2 detected as UNSAFE.")

if __name__ == "__main__":
    test_archer_check()
    test_bomb_check()
    test_trailblazer_check()
