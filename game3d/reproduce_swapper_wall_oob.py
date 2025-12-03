
import numpy as np
import logging
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.pieces.pieces.swapper import generate_swapper_moves
from game3d.game.turnmove import make_move

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reproduce_swapper_wall_oob():
    print("--- Reproducing Swapper-Wall OOB Issue ---")
    
    # 1. Setup GameState
    from game3d.board.board import Board
    board = Board()
    state = GameState(board, Color.WHITE)
    
    # Clear board (cache is source of truth)
    # We can just initialize a new empty board state or clear the cache
    # Assuming cache is already initialized empty by GameState() + Board()
    # But GameState() initializes from Board.startpos() usually?
    # No, GameState(board) uses board.get_initial_setup() if board is not None?
    # Actually GameState.__init__ calls cache_manager.initialize_from_board(board)
    
    # To clear the board, we can just clear the cache
    # But cache_manager might not have a clear() method exposed easily.
    # Let's just overwrite the positions we need.
    # Or better, create an empty board.
    
    # 2. Place a Swapper at [7, 8, 8] (Invalid for Wall, Valid for Swapper)
    swapper_pos = np.array([7, 8, 8], dtype=np.int16)
    # We don't need to set board.grid
    state.cache_manager.occupancy_cache.set_position(swapper_pos, (PieceType.SWAPPER, Color.WHITE))
    
    # 3. Place a Wall at [5, 5, 5] (Valid for Wall)
    wall_pos = np.array([5, 5, 5], dtype=np.int16)
    # Wall occupies 2x2
    state.cache_manager.occupancy_cache.set_position(wall_pos, (PieceType.WALL, Color.WHITE))
    state.cache_manager.occupancy_cache.set_position(np.array([6, 5, 5]), (PieceType.WALL, Color.WHITE))
    state.cache_manager.occupancy_cache.set_position(np.array([5, 6, 5]), (PieceType.WALL, Color.WHITE))
    state.cache_manager.occupancy_cache.set_position(np.array([6, 6, 5]), (PieceType.WALL, Color.WHITE))
    
    print(f"Swapper at {swapper_pos}")
    print(f"Wall at {wall_pos}")
    
    # 4. Generate moves for Swapper
    moves = generate_swapper_moves(state.cache_manager, Color.WHITE, swapper_pos)
    
    # 5. Check if swap with Wall is generated
    swap_move = None
    for i in range(moves.shape[0]):
        # Check if destination is the Wall anchor
        if np.array_equal(moves[i, 3:], wall_pos):
            swap_move = moves[i]
            break
            
    if swap_move is not None:
        print(f"❌ Swapper generated move to Wall: {swap_move}")
        print("This move will place the Wall at [7, 8, 8], which is OUT OF BOUNDS for a Wall.")
        
        # 6. Try to execute the move
        try:
            print("Attempting to execute swap...")
            make_move(state, swap_move)
            print("❌ Move executed successfully! Wall is now at [7, 8, 8].")
            
            # Verify Wall is at [7, 8, 8]
            piece_at_swapper_start = state.cache_manager.occupancy_cache.get(swapper_pos)
            if piece_at_swapper_start and piece_at_swapper_start['piece_type'] == PieceType.WALL:
                print("Confirmed: Wall is at [7, 8, 8].")
                
                # Check if it's OOB
                # Wall at [7, 8, 8] occupies (7,8), (8,8), (7,9), (8,9)
                # y=9 is OOB.
                print("Wall at [7, 8, 8] is effectively OOB.")
                
        except Exception as e:
            print(f"✅ Move execution failed as expected: {e}")
            
    else:
        print("✅ Swapper did NOT generate move to Wall.")

if __name__ == "__main__":
    reproduce_swapper_wall_oob()
