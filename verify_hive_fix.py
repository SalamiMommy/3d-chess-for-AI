
import numpy as np
from game3d.game.factory import start_game_state
from game3d.pieces.pieces.hive import apply_multi_hive_move
from game3d.movement.movepiece import Move
from game3d.common.shared_types import PieceType, Color

def test_hive_move_turn_count():
    print("Initializing game state...")
    state = start_game_state()
    
    # Setup: Place a Hive for White
    hive_pos = np.array([4, 4, 4])
    state.cache_manager.occupancy_cache.set_position(hive_pos, np.array([PieceType.HIVE, Color.WHITE]))
    state.board.set_piece_at(hive_pos, PieceType.HIVE, Color.WHITE)
    
    initial_turn = state.turn_number
    initial_clock = state.halfmove_clock
    print(f"Initial Turn: {initial_turn}, Clock: {initial_clock}")
    
    # Simulate a Hive move
    target_pos = np.array([5, 4, 4])
    move = Move(hive_pos, target_pos)
    
    print(f"Applying Hive move from {hive_pos} to {target_pos}...")
    new_state = apply_multi_hive_move(state, move)
    
    new_turn = new_state.turn_number
    new_clock = new_state.halfmove_clock
    print(f"New Turn: {new_turn}, Clock: {new_clock}")
    
    if new_turn == initial_turn and new_clock == initial_clock:
        print("SUCCESS: Turn counters remained constant during intermediate Hive move.")
    else:
        print(f"FAILURE: Turn counters changed! Expected ({initial_turn}, {initial_clock}), got ({new_turn}, {new_clock})")
        exit(1)

if __name__ == "__main__":
    test_hive_move_turn_count()
