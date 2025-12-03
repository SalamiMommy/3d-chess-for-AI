
import unittest
import logging
import numpy as np
from io import StringIO
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.game.terminal import is_game_over

class TestCheckmateLogging(unittest.TestCase):
    def setUp(self):
        # Capture logs
        self.log_capture = StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.logger = logging.getLogger('game3d.game.terminal')
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    def test_checkmate_logging(self):
        # Setup a simple checkmate: White King at (0,0,0), Black Rook at (0,0,5) attacking it
        # And White King has no moves.
        # Wait, King at (0,0,0) might have moves to (0,1,0), (1,0,0), etc.
        # I need to block escape squares or use more pieces.
        
        # Easiest checkmate: Fool's mate style or just surround the king.
        # Let's try to construct a state where White is checkmated.
        
        # Initialize GameState with empty board
        board = Board()
        game_state = GameState(board, Color.WHITE)
        
        # Define pieces for checkmate scenario
        # White King at (0,0,0)
        # Black pieces surrounding it
        coords = [
            [0,0,0], # White King
            [0,0,5], # Black Rook
            [0,5,0], # Black Rook
            [5,0,0], # Black Rook
            [0,0,2], # Black Queen
            [0,2,0], # Black Queen
            [2,0,0], # Black Queen
            [2,2,2], # Black Queen
            [2,2,0], # Black Queen
            [2,0,2], # Black Queen
            [0,2,2], # Black Queen
            [1,1,5]  # Black Rook (Attacks [1,1,1])
        ]
        
        types = [
            PieceType.KING,
            PieceType.ROOK,
            PieceType.ROOK,
            PieceType.ROOK,
            PieceType.QUEEN,
            PieceType.QUEEN,
            PieceType.QUEEN,
            PieceType.QUEEN,
            PieceType.QUEEN,
            PieceType.QUEEN,
            PieceType.QUEEN,
            PieceType.ROOK
        ]
        
        colors = [
            Color.WHITE,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK,
            Color.BLACK
        ]
        
        # Convert to numpy arrays
        coords_arr = np.array(coords, dtype=np.int32) # COORD_DTYPE is usually int32 or int8
        types_arr = np.array([t.value for t in types], dtype=np.int8)
        colors_arr = np.array(colors, dtype=np.int8)
        
        # Rebuild cache with this state
        if game_state.cache_manager:
            game_state.cache_manager.occupancy_cache.rebuild(coords_arr, types_arr, colors_arr)
            
        # Check if game over
        result = is_game_over(game_state)
        
        # Check logs
        logs = self.log_capture.getvalue()
        print("Captured Logs:\n", logs)
        
        self.assertTrue(result, "Game should be over")
        self.assertIn("Game over: Checkmate (Winner: BLACK, Turn: 1)", logs)
        self.assertIn("White: Priests=0, King=(0,0,0)", logs)
        self.assertIn("Attackers:", logs)
        self.assertIn("QUEEN at", logs)

if __name__ == '__main__':
    unittest.main()
