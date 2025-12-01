import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE
from game3d.game.turnmove import execute_hive_move
from game3d.movement.movepiece import Move
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager

def test_hive_turn_mechanic():
    # 1. Setup empty board
    board = Board.empty()
    
    empty_coords = np.empty((0, 3), dtype=COORD_DTYPE)
    empty_types = np.empty(0, dtype=PIECE_TYPE_DTYPE)
    empty_colors = np.empty(0, dtype=COLOR_DTYPE)
    
    cache = OptimizedCacheManager(board, Color.WHITE, initial_data=(empty_coords, empty_types, empty_colors))
    state = GameState(board=board, color=Color.WHITE, cache_manager=cache)
    
    # 2. Place 2 Hives for White
    # Hive 1 at (0, 0, 0)
    # Hive 2 at (2, 0, 0)
    h1_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    h2_pos = np.array([2, 0, 0], dtype=COORD_DTYPE)
    
    cache.occupancy_cache.set_position(h1_pos, (PieceType.HIVE, Color.WHITE))
    cache.occupancy_cache.set_position(h2_pos, (PieceType.HIVE, Color.WHITE))
    
    # Place a King for White (needed for valid state usually)
    k_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(k_pos, (PieceType.KING, Color.WHITE))
    
    # Place a King for Black
    bk_pos = np.array([8, 8, 8], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(bk_pos, (PieceType.KING, Color.BLACK))
    
    # Ensure generator is initialized
    from game3d.movement import generator
    generator.initialize_generator()
    generator._generator.update_legal_moves_incremental(state)
    
    print(f"Initial Turn: {state.color}")
    assert state.color == Color.WHITE
    
    # 3. Move Hive 1
    # Hive moves like a King (1 step in any direction)
    h1_dest = np.array([0, 1, 0], dtype=COORD_DTYPE)
    move1 = Move(h1_pos, h1_dest)
    
    print(f"Executing Move 1: {move1}")
    # We must use the game3d.py logic which delegates to turnmove.execute_hive_move
    # But since we are testing turnmove directly:
    new_state = execute_hive_move(state, move1)
    
    print(f"Turn after Move 1: {new_state.color}")
    
    # Verify turn is still White
    if new_state.color != Color.WHITE:
        print("FAIL: Turn switched after first hive move, but another hive is available")
        return False
        
    # Verify Hive 1 moved
    p1 = new_state.cache_manager.occupancy_cache.get(h1_dest)
    assert p1 is not None and p1['piece_type'] == PieceType.HIVE
    
    # Verify Hive 2 hasn't moved
    p2 = new_state.cache_manager.occupancy_cache.get(h2_pos)
    assert p2 is not None and p2['piece_type'] == PieceType.HIVE
    
    # 4. Move Hive 2
    h2_dest = np.array([2, 1, 0], dtype=COORD_DTYPE)
    move2 = Move(h2_pos, h2_dest)
    
    print(f"Executing Move 2: {move2}")
    final_state = execute_hive_move(new_state, move2)
    
    print(f"Turn after Move 2: {final_state.color}")
    
    # Verify turn switched to Black
    if final_state.color != Color.BLACK:
        print("FAIL: Turn did not switch after all hives moved")
        return False
        
    print("PASS: Hive turn mechanic verified")
    return True

if __name__ == "__main__":
    if test_hive_turn_mechanic():
        sys.exit(0)
    else:
        sys.exit(1)
