
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
        
        board = Board(SIZE)
        # Clear board
        board.clear()
        
        # Place White King at (0,0,0)
        board.add_piece(PieceType.KING, Color.WHITE, (0,0,0))
        
        # Place Black Rook at (0,0,5) - attacks (0,0,0)
        board.add_piece(PieceType.ROOK, Color.BLACK, (0,0,5))
        
        # Place Black Rook at (0,5,0) - attacks (0,0,0) and cuts off y-axis
        board.add_piece(PieceType.ROOK, Color.BLACK, (0,5,0))
        
        # Place Black Rook at (5,0,0) - attacks (0,0,0) and cuts off x-axis
        board.add_piece(PieceType.ROOK, Color.BLACK, (5,0,0))
        
        # This might not be enough to cover diagonal escapes (1,1,1) etc.
        # Let's just surround the king with Black Queens.
        
        # White King at (0,0,0)
        # Black Queen at (0,0,2) -> attacks (0,0,0) and (0,0,1)
        # Black Queen at (0,2,0) -> attacks (0,0,0) and (0,1,0)
        # Black Queen at (2,0,0) -> attacks (0,0,0) and (1,0,0)
        # Black Queen at (2,2,2) -> attacks (0,0,0) and (1,1,1)
        
        board.add_piece(PieceType.QUEEN, Color.BLACK, (0,0,2))
        board.add_piece(PieceType.QUEEN, Color.BLACK, (0,2,0))
        board.add_piece(PieceType.QUEEN, Color.BLACK, (2,0,0))
        board.add_piece(PieceType.QUEEN, Color.BLACK, (2,2,2))
        
        # Also need to block (1,1,0), (1,0,1), (0,1,1)
        board.add_piece(PieceType.QUEEN, Color.BLACK, (2,2,0))
        board.add_piece(PieceType.QUEEN, Color.BLACK, (2,0,2))
        board.add_piece(PieceType.QUEEN, Color.BLACK, (0,2,2))
        
        # Initialize GameState
        game_state = GameState(board, Color.WHITE)
        
        # Ensure cache is built
        if game_state.cache_manager:
            game_state.cache_manager.rebuild(board)
            
        # Check if game over
        result = is_game_over(game_state)
        
        # Check logs
        logs = self.log_capture.getvalue()
        print("Captured Logs:\n", logs)
        
        self.assertTrue(result, "Game should be over")
        self.assertIn("Game over: Checkmate (Winner: BLACK)", logs)
        self.assertIn("White: Priests=0, King=(0,0,0)", logs)
        self.assertIn("Attackers:", logs)
        self.assertIn("QUEEN at", logs)

if __name__ == '__main__':
    unittest.main()
