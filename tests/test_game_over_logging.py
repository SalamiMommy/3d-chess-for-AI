
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from game3d.game import terminal
from game3d.common.shared_types import Color, PieceType, MOVE_DTYPE

class TestGameOverLogging(unittest.TestCase):
    @patch('game3d.game.terminal.logger')
    def test_stalemate_logging_with_priests_and_kings(self, mock_logger):
        # Setup mock game state
        game_state = MagicMock()
        game_state.color = Color.BLACK.value
        game_state.legal_moves = np.array([], dtype=MOVE_DTYPE) # No legal moves
        game_state.halfmove_clock = 0
        game_state.zkey = 123
        
        # Mock cache manager and occupancy cache
        cache_manager = MagicMock()
        occ_cache = MagicMock()
        cache_manager.occupancy_cache = occ_cache
        game_state.cache_manager = cache_manager
        
        # Setup occupancy cache returns
        # Mock get_all_occupied_vectorized to return some pieces for Black (blocked)
        # coords, piece_types, colors
        # We need piece_count > 1 for "Material Blocked" case
        colors = np.array([Color.BLACK.value, Color.BLACK.value, Color.WHITE.value])
        occ_cache.get_all_occupied_vectorized.return_value = (None, None, colors)
        
        # Mock priest counts
        occ_cache.get_priest_count.side_effect = lambda c: 1 if c == Color.WHITE.value else 0
        
        # Mock king positions
        occ_cache.find_king.side_effect = lambda c: np.array([0, 0, 0]) if c == Color.WHITE.value else np.array([8, 8, 8])
        
        # Mock is_check to return False (Stalemate)
        with patch('game3d.game.terminal.is_check', return_value=False):
            with patch('game3d.game.terminal.is_repetition_draw', return_value=False):
                with patch('game3d.game.terminal.is_move_rule_draw', return_value=False):
                    with patch('game3d.game.terminal.is_insufficient_material', return_value=False):
                        # Run is_game_over
                        terminal.is_game_over(game_state)
        
        # Verify logger was called with expected message
        # We expect "Material Blocked" because Black has 2 pieces (from colors array)
        # Priests: W=1, B=0
        # Kings: W=(0,0,0), B=(8,8,8)
        
        expected_fragment = "Priests: W=1, B=0 | Kings: W=(0,0,0), B=(8,8,8)"
        
        # Check if any warning call contains the fragment
        found = False
        for call in mock_logger.warning.call_args_list:
            args, _ = call
            if expected_fragment in args[0]:
                found = True
                break
        
        self.assertTrue(found, f"Expected log fragment '{expected_fragment}' not found in logger calls")

if __name__ == '__main__':
    unittest.main()
