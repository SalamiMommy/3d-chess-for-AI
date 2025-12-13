
import unittest
import numpy as np
import logging
from game3d.game.factory import start_game_state
from game3d.common.shared_types import PieceType, Color
from game3d.game.turnmove import validate_move_integrated, legal_moves
from game3d.movement.movepiece import Move

# Configure logging to error only
logging.basicConfig(level=logging.ERROR)

class TestKingCaptureRules(unittest.TestCase):
    def setUp(self):
        # Setup consistent board
        self.state = start_game_state(ensure_start_pos=False)
        self.cache = self.state.cache_manager.occupancy_cache
        self.state.cache_manager.move_cache.invalidate()

    def test_capture_attempt_no_priest(self):
        print("\n--- Test: Can King be captured (No Priest)? ---")
        
        # White King at (0,0,0) - NO PRIEST
        self.cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE], dtype=np.int8))
        
        # Black King at (7,7,7) (Safe)
        self.cache.set_position(np.array([7,7,7]), np.array([PieceType.KING, Color.BLACK], dtype=np.int8))
        
        # Black Rook at (0,0,5)
        self.cache.set_position(np.array([0,0,5]), np.array([PieceType.ROOK, Color.BLACK], dtype=np.int8))
        
        self.state.color = Color.BLACK
        
        moves = legal_moves(self.state)
        found = any(np.array_equal(m[:3], [0,0,5]) and np.array_equal(m[3:6], [0,0,0]) for m in moves)
        
        if found:
            print("RESULT: King capture is ALLOWED (Correct).")
        else:
            print("RESULT: King capture is PREVENTED (Incorrect - should be allowed if no priest).")
            # Debug: why filtered?
            # Manually check generator or validation if needed
            self.fail("King should be capturable when no priests are present!")

    def test_capture_with_priest(self):
        print("\n--- Test: Can King be captured WITH Priest? ---")
        
        # White King at (0,0,0) WITH Priest
        self.cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE], dtype=np.int8))
        self.cache.set_position(np.array([1,1,1]), np.array([PieceType.PRIEST, Color.WHITE], dtype=np.int8))
        
        # Black King (Safe)
        self.cache.set_position(np.array([7,7,7]), np.array([PieceType.KING, Color.BLACK], dtype=np.int8))

        self.cache.set_position(np.array([0,0,5]), np.array([PieceType.ROOK, Color.BLACK], dtype=np.int8))

        self.state.color = Color.BLACK
        
        moves = legal_moves(self.state)
        found = any(np.array_equal(m[:3], [0,0,5]) and np.array_equal(m[3:6], [0,0,0]) for m in moves)
        
        if found:
            print("RESULT WITH PRIEST: King capture allowed (Incorrect).")
            self.fail("King capture allowed despite Priest!")
        else:
            print("RESULT WITH PRIEST: King capture prevented (Correct).")

if __name__ == "__main__":
    unittest.main()
