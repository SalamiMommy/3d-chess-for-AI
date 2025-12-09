
import unittest
import logging
import numpy as np
from unittest.mock import MagicMock, patch

from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType

class TestCacheDesyncDetection(unittest.TestCase):
    def setUp(self):
        # Suppress logging during tests
        logging.getLogger('game3d').setLevel(logging.ERROR)
        
    def test_detection_of_occ_type_mismatch(self):
        """Verify that validate_consistency detects mismatch."""
        state = GameState.from_startpos()
        cache = state.cache_manager.occupancy_cache
        
        # Clear board
        cache.clear()
        
        # Manually corrupt: Color=White but Type=0 (Empty) at (0,0,0)
        cache._occ[0, 0, 0] = Color.WHITE
        cache._ptype[0, 0, 0] = 0
        
        is_valid, msg = cache.validate_consistency()
        print(f"Validation Msg: {msg}")
        
        self.assertFalse(is_valid)
        self.assertIn("mismatched occupancy/type", msg)
        self.assertIn("Color=1, Type=0", msg)

    def test_set_position_enforces_consistency(self):
        """Verify that set_position raises ValueError on inconsistent input."""
        state = GameState.from_startpos()
        cache = state.cache_manager.occupancy_cache
        
        # Try to set inconsistent state: Color but no Type
        coord = np.array([0, 0, 0], dtype=np.int16)
        piece = np.array([0, Color.WHITE], dtype=np.int8) # Type 0, Color 1
        
        with self.assertRaises(ValueError) as cm:
            cache.set_position(coord, piece)
        self.assertIn("Inconsistent SetPosition", str(cm.exception))

    def test_detection_in_move_cache(self):
        """Verify that MoveCache checks consistency when no moves found."""
        state = GameState.from_startpos()
        cache = state.cache_manager.occupancy_cache
        
        # Clear board -> 0 legal moves naturally
        # But we need "piece_count > 0" to trigger the check
        cache.clear()
        
        # Add a "Ghost" piece (Color=White, Type=0)
        # This triggers piece_count=1 (from get_positions loop over _occ)
        # But generator generates 0 moves because Type=0 usually means empty/invalid for generator.
        cache._occ[0, 0, 0] = Color.WHITE
        cache._ptype[0, 0, 0] = 0
        
        # We need to ensure state.color is WHITE
        state.color = Color.WHITE
        
        # Patch logger to verify call
        with patch('game3d.cache.caches.movecache.logger') as mock_logger:
            # Trigger generation
            # This calls generate_legal_moves -> generate_fused -> store_legal_moves
            from game3d.movement.generator import generate_legal_moves
            moves = generate_legal_moves(state)
            
            # Assert moves are empty
            self.assertEqual(moves.size, 0)
            
            # Assert logger.error was called with "CACHE DESYNC DETECTED"
            mock_logger.error.assert_called()
            args, _ = mock_logger.error.call_args
            self.assertIn("CACHE DESYNC DETECTED", args[0])
            print("Successfully triggered CACHE DESYNC log")

if __name__ == '__main__':
    unittest.main()
