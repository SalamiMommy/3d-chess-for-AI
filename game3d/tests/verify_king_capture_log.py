
import numpy as np
import logging
from game3d.game.gamestate import GameState
from game3d.game.turnmove import make_move_trusted
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("game3d.game.turnmove")
logger.setLevel(logging.INFO) # Ensure we capture the critical log

def test_king_capture_log():
    print("--- Starting King Capture Log Verification ---")
    
    # 1. Setup GameState
    state = GameState.from_startpos()
    
    # 2. Setup a scenario: White Rook next to Black King
    # Clear board for simplicity
    state.cache_manager.occupancy_cache.clear_all()
    
    # Place Black King at (0, 0, 0)
    state.cache_manager.occupancy_cache.set_position(
        np.array([0, 0, 0], dtype=COORD_DTYPE), 
        np.array([PieceType.KING, Color.BLACK])
    )
    
    # Place White Rook at (0, 0, 1)
    state.cache_manager.occupancy_cache.set_position(
        np.array([0, 0, 1], dtype=COORD_DTYPE), 
        np.array([PieceType.ROOK, Color.WHITE])
    )
    
    # 3. Create a move: Rook captures King
    # Move: (0, 0, 1) -> (0, 0, 0)
    move = np.array([0, 0, 1, 0, 0, 0], dtype=COORD_DTYPE)
    
    print("Attempting to capture King with make_move_trusted...")
    
    try:
        # Use make_move_trusted to bypass legality checks
        make_move_trusted(state, move)
        print("Move executed.")
    except Exception as e:
        print(f"Move execution failed: {e}")

    print("--- Verification Finished ---")

if __name__ == "__main__":
    test_king_capture_log()
