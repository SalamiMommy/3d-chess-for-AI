
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.game.turnmove import make_move

def test_freeze_logic():
    print("Initializing GameState...")
    board = Board()
    state = GameState(board, Color.WHITE)
    
    # Setup:
    # White Freezer at (2, 2, 2)
    # Black Pawn at (2, 2, 3) (Adjacent, should be frozen)
    # White Pawn at (0, 0, 0) (To make a move)
    
    freezer_pos = np.array([2, 2, 2])
    enemy_pos = np.array([2, 2, 3])
    friendly_mover_pos = np.array([0, 0, 0])
    friendly_mover_dest = np.array([0, 0, 1])
    
    print(f"Placing White Freezer at {freezer_pos}")
    state.cache_manager.occupancy_cache.set_position(freezer_pos, np.array([PieceType.FREEZER, Color.WHITE]))
    
    print(f"Placing Black Pawn at {enemy_pos}")
    state.cache_manager.occupancy_cache.set_position(enemy_pos, np.array([PieceType.PAWN, Color.BLACK]))
    
    print(f"Placing White Pawn at {friendly_mover_pos}")
    state.cache_manager.occupancy_cache.set_position(friendly_mover_pos, np.array([PieceType.PAWN, Color.WHITE]))
    
    # Verify initial state
    print("\n--- Initial State (Turn 1, White) ---")
    # Check if Black Pawn is frozen (Should NOT be frozen yet, as no move made)
    # But wait, we are checking state.turn_number (1).
    # Expiry is 0. 0 < 1. Not frozen.
    is_frozen = state.cache_manager.consolidated_aura_cache.batch_is_frozen(
        enemy_pos.reshape(1, 3), state.turn_number, Color.BLACK
    )[0]
    print(f"Black Pawn Frozen: {is_frozen} (Expected: False)")
    
    # Make a move with White Pawn
    print(f"\n--- White Moves Pawn {friendly_mover_pos} -> {friendly_mover_dest} ---")
    move = np.concatenate([friendly_mover_pos, friendly_mover_dest])
    new_state = make_move(state, move)
    
    print(f"New Turn: {new_state.turn_number} (Color: {new_state.color})")
    
    # Check if Black Pawn is frozen in new state (Turn 2, Black)
    # Expiry should be 1 + 1 = 2.
    # Turn is 2.
    # 2 <= 2 -> Frozen.
    is_frozen_new = new_state.cache_manager.consolidated_aura_cache.batch_is_frozen(
        enemy_pos.reshape(1, 3), new_state.turn_number, Color.BLACK
    )[0]
    print(f"Black Pawn Frozen: {is_frozen_new} (Expected: True)")
    
    # Verify Black cannot move the frozen pawn
    print("\n--- Generating Moves for Black ---")
    from game3d.movement.generator import generate_legal_moves
    moves = generate_legal_moves(new_state)
    
    # Check if any move starts from enemy_pos
    pawn_moves = moves[np.all(moves[:, :3] == enemy_pos, axis=1)]
    print(f"Moves for Black Pawn: {len(pawn_moves)} (Expected: 0)")
    
    if len(pawn_moves) == 0 and is_frozen_new:
        print("SUCCESS: Black Pawn is frozen and cannot move.")
    else:
        print("FAILURE: Black Pawn moved or was not frozen.")
        
    # Simulate Black passing (or making another move if they had other pieces)
    # Since Black has no other pieces, they might be stalemated?
    # Let's add another Black piece far away that is NOT frozen.
    far_enemy_pos = np.array([7, 7, 7])
    print(f"\nPlacing Far Black Pawn at {far_enemy_pos}")
    new_state.cache_manager.occupancy_cache.set_position(far_enemy_pos, np.array([PieceType.PAWN, Color.BLACK]))
    
    # Regenerate moves
    moves = generate_legal_moves(new_state)
    far_pawn_moves = moves[np.all(moves[:, :3] == far_enemy_pos, axis=1)]
    print(f"Moves for Far Black Pawn: {len(far_pawn_moves)} (Expected: > 0)")
    
    if len(far_pawn_moves) > 0:
        # Make a move with far pawn to advance turn
        move_far = far_pawn_moves[0]
        print(f"Black moves Far Pawn: {move_far[:3]} -> {move_far[3:]}")
        state_turn_3 = make_move(new_state, move_far)
        
        print(f"\n--- Turn 3 (White) ---")
        # Check if Black Pawn is still frozen?
        # Turn is 3. Expiry was 2.
        # 2 < 3 -> Not frozen.
        # But wait, we are checking `batch_is_frozen` with `state_turn_3.turn_number` (3).
        # And `victim_color` is BLACK.
        # So we check if Black Pawn is frozen.
        is_frozen_turn_3 = state_turn_3.cache_manager.consolidated_aura_cache.batch_is_frozen(
            enemy_pos.reshape(1, 3), state_turn_3.turn_number, Color.BLACK
        )[0]
        print(f"Black Pawn Frozen: {is_frozen_turn_3} (Expected: False)")
        
        if not is_frozen_turn_3:
            print("SUCCESS: Freeze expired correctly.")
        else:
            print("FAILURE: Freeze did not expire.")

if __name__ == "__main__":
    test_freeze_logic()
