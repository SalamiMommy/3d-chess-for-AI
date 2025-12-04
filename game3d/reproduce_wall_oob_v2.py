
import numpy as np
import logging
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.pieces.pieces.wall import generate_wall_moves
from game3d.board.board import Board
from game3d.game.turnmove import make_move

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reproduce_wall_oob():
    print("--- Reproducing Wall OOB Issue ---")
    
    # 1. Setup GameState
    board = Board()
    state = GameState(board, Color.WHITE)
    
    # Clear neighbors to ensure [6, 7, 1] is an anchor
    state.cache_manager.occupancy_cache.set_position(np.array([5, 7, 1]), None)
    state.cache_manager.occupancy_cache.set_position(np.array([6, 6, 1]), None)
    
    # Clear destination area to ensure movement is possible
    for x in range(4, 8):
        for y in range(6, 9):
            for z in range(0, 4):
                if not (x==6 and y==7 and z==1) and \
                   not (x==7 and y==7 and z==1) and \
                   not (x==6 and y==8 and z==1) and \
                   not (x==7 and y==8 and z==1):
                    state.cache_manager.occupancy_cache.set_position(np.array([x, y, z]), None)
    
    # 2. Place a Wall at [6, 7, 1]
    wall_pos = np.array([6, 7, 1], dtype=np.int16)
    
    # Set the 4 squares for the wall
    state.cache_manager.occupancy_cache.set_position(wall_pos, (PieceType.WALL, Color.WHITE))
    state.cache_manager.occupancy_cache.set_position(np.array([7, 7, 1]), (PieceType.WALL, Color.WHITE))
    state.cache_manager.occupancy_cache.set_position(np.array([6, 8, 1]), (PieceType.WALL, Color.WHITE))
    state.cache_manager.occupancy_cache.set_position(np.array([7, 8, 1]), (PieceType.WALL, Color.WHITE))
    
    print(f"Wall placed at {wall_pos}")
    
    # 3. Generate moves for Wall
    print("Generating moves...")
    moves = generate_wall_moves(state.cache_manager, Color.WHITE, wall_pos)
    
    print(f"Generated {moves.shape[0]} moves.")
    
    found = False
    target_pos = np.array([5, 8, 2], dtype=np.int16)
    for i in range(moves.shape[0]):
        dest = moves[i, 3:]
        if np.array_equal(dest, target_pos):
            found = True
            print(f"❌ FOUND INVALID MOVE: {moves[i]}")
            break
            
    if not found:
        print("✅ Did NOT generate the invalid move to [5, 8, 2].")

    # 4. Manually inject the invalid move to verify error message
    print("\nInjecting invalid move [6, 7, 1] -> [5, 8, 2]...")
    invalid_move = np.array([6, 7, 1, 5, 8, 2], dtype=np.int16)
    
    try:
        make_move(state, invalid_move)
        print("❌ Move executed successfully (Unexpected!)")
    except Exception as e:
        print(f"✅ Move rejected as expected. Error: {e}")

if __name__ == "__main__":
    reproduce_wall_oob()
