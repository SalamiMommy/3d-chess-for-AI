
import sys
import os
import logging
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.game.turnmove import _log_move_if_needed
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color

def test_move_reporting():
    # Mock logger
    with patch('game3d.game.turnmove.logger') as mock_logger:
        # Create a dummy game state
        # We only need turn_number and history for this test
        game_state = MagicMock(spec=GameState)
        game_state.history = [] # Empty history
        
        # Test case 1: Turn 100 (should log)
        game_state.turn_number = 100
        mv = np.array([0,0,0, 1,1,1])
        from_piece = {"piece_type": PieceType.PAWN, "color": Color.WHITE}
        captured_piece = None
        
        _log_move_if_needed(game_state, from_piece, captured_piece, mv)
        
        if mock_logger.info.called:
            print("SUCCESS: Logged at turn 100")
        else:
            print("FAILURE: Did not log at turn 100")
            
        mock_logger.reset_mock()
        
        # Test case 2: Turn 101 (should NOT log)
        game_state.turn_number = 101
        _log_move_if_needed(game_state, from_piece, captured_piece, mv)
        
        if not mock_logger.info.called:
            print("SUCCESS: Did not log at turn 101")
        else:
            print("FAILURE: Logged at turn 101")

        mock_logger.reset_mock()

        # Test case 3: Turn 200 (should log)
        game_state.turn_number = 200
        _log_move_if_needed(game_state, from_piece, captured_piece, mv)
        
        if mock_logger.info.called:
            print("SUCCESS: Logged at turn 200")
        else:
            print("FAILURE: Did not log at turn 200")

if __name__ == "__main__":
    test_move_reporting()
