
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color, SIZE
from game3d.core.move_logic import calculate_move_effects

from game3d.core.buffer import state_to_buffer
from game3d.attacks.check import square_attacked_by

def test_bomb_detonation_king_immunity():
    print("\n--- Test 1: Bomb Detonation vs King (No Priest) ---")
    game = GameState.from_startpos()
    
    # Clear board
    # Clear board
    occ = game.cache_manager.occupancy_cache
    occ._ptype.fill(0)
    occ._occ.fill(0)
    occ._priest_count.fill(0)
    occ._king_positions.fill(-1)
    occ._positions_dirty = [True, True]
    

    occ.set_position(np.array([4,4,4], dtype=np.int16), np.array([PieceType.BOMB, Color.WHITE]))
    occ.set_position(np.array([4,4,5], dtype=np.int16), np.array([PieceType.KING, Color.BLACK]))
    
    # Simulate Bomb "Wait" move (self-move)
    move = np.array([4,4,4, 4,4,4], dtype=np.int16)
    
    buffer = state_to_buffer(game)
    effects = calculate_move_effects(move, buffer)
    
    # Check if King is in cleared coords
    king_pos = np.array([4,4,5])
    king_cleared = False
    for coord in effects.coords_to_clear:
        if np.array_equal(coord, king_pos):
            king_cleared = True
            break
            
    print(f"King cleared: {king_cleared}")
    if king_cleared:
        print("PASS: King destroyed (No Priest)")
    else:
        print("FAIL: King survived (No Priest)")

def test_bomb_detonation_king_immunity_with_priest():
    print("\n--- Test 2: Bomb Detonation vs King (With Priest) ---")
    game = GameState.from_startpos()
    
    # Clear board
    # Clear board
    occ = game.cache_manager.occupancy_cache
    occ._ptype.fill(0)
    occ._occ.fill(0)
    occ._priest_count.fill(0)
    occ._king_positions.fill(-1)
    occ._positions_dirty = [True, True]
    
    # Setup: White Bomb at (4,4,4), Black King at (4,4,5)
    # ADD Black Priest at (0,0,0)
    
    occ.set_position(np.array([4,4,4], dtype=np.int16), np.array([PieceType.BOMB, Color.WHITE]))
    occ.set_position(np.array([4,4,5], dtype=np.int16), np.array([PieceType.KING, Color.BLACK]))
    
    occ.set_position(np.array([0,0,0], dtype=np.int16), np.array([PieceType.PRIEST, Color.BLACK]))
    
    # Simulate Bomb "Wait" move (self-move)
    move = np.array([4,4,4, 4,4,4], dtype=np.int16)
    
    buffer = state_to_buffer(game)
    effects = calculate_move_effects(move, buffer)
    
    # Check if King is in cleared coords
    king_pos = np.array([4,4,5])
    king_cleared = False
    for coord in effects.coords_to_clear:
        if np.array_equal(coord, king_pos):
            king_cleared = True
            break
            
    print(f"King cleared: {king_cleared}")
    if not king_cleared:
        print("PASS: King survived (With Priest)")
    else:
        print("FAIL: King destroyed (With Priest)")

def test_bomb_check_radius():
    print("\n--- Test 3: Bomb Check Radius 2 ---")
    game = GameState.from_startpos()
    
    # Clear board
    # Clear board
    occ = game.cache_manager.occupancy_cache
    occ._ptype.fill(0)
    occ._occ.fill(0)
    occ._priest_count.fill(0)
    occ._king_positions.fill(-1)
    occ._positions_dirty = [True, True]
    
    # Setup: White Bomb at (4,4,4), Black King at (4,4,6) (Distance 2)
    # Should be Check if Radius 2 implemented
    
    occ.set_position(np.array([4,4,4], dtype=np.int16), np.array([PieceType.BOMB, Color.WHITE]))
    occ.set_position(np.array([4,4,6], dtype=np.int16), np.array([PieceType.KING, Color.BLACK]))
    game.cache_manager.move_cache.invalidate()
    game.cache_manager._zkey = game.cache_manager._compute_initial_zobrist(game.color)
    
    # Check black king status
    # Game is White to move, but we want to see if Black King is in check?
    # No, usually we check if the player WHOSE TURN IT IS is in check.
    # Set turn to Black
    game.color = Color.BLACK
    
    king_pos = np.array([4,4,6], dtype=np.int16)
    in_check = square_attacked_by(game.board, Color.BLACK, king_pos, Color.WHITE.value, game.cache_manager)
    print(f"Black King in check: {in_check}")
    
    if in_check:
        print("PASS: King in check at Radius 2")
    else:
        print("FAIL: King NOT in check at Radius 2")


if __name__ == "__main__":
    test_bomb_detonation_king_immunity()
    test_bomb_detonation_king_immunity_with_priest()
    test_bomb_check_radius()
