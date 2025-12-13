
import unittest
import numpy as np
import logging
from game3d.game.factory import start_game_state
from game3d.common.shared_types import PieceType, Color
from game3d.game.turnmove import validate_move_integrated, legal_moves

# Configure logging to error only
logging.basicConfig(level=logging.ERROR)

class TestSwapperKing(unittest.TestCase):
    def setUp(self):
        self.state = start_game_state(ensure_start_pos=False)
        self.cache = self.state.cache_manager.occupancy_cache
        self.state.cache_manager.move_cache.invalidate()

    def test_swapper_blocked_by_opponent_priest(self):
        print("\n--- Test: Swapper Swap vs Opponent Priest ---")
        
        # Setup: White King (0,0,0), White Swapper (0,0,5)
        self.cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE], dtype=np.int8))
        self.cache.set_position(np.array([0,0,5]), np.array([PieceType.SWAPPER, Color.WHITE], dtype=np.int8))
        
        # ADD BLACK PRIEST (Protects Black King, but shouldn't affect White actions vs White King)
        self.cache.set_position(np.array([7,7,7]), np.array([PieceType.PRIEST, Color.BLACK], dtype=np.int8))
        
        self.state.color = Color.WHITE
        
        # Check moves
        moves = legal_moves(self.state)
        found = False
        for m in moves:
            if np.array_equal(m[:3], [0,0,5]) and np.array_equal(m[3:6], [0,0,0]):
                found = True
                break
        
        if found:
            print("RESULT: Swap ALLOWED (Correct).")
        else:
            print("RESULT: Swap BLOCKED (Bug: Opponent priest shouldn't block friendly swap).")
            self.fail("Friendly swap blocked by opponent priest logic!")

if __name__ == "__main__":
    unittest.main()
