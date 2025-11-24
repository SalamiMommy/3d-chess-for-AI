
import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color, MOVE_DTYPE, COORD_DTYPE
from game3d.game.turnmove import legal_moves, make_move, filter_safe_moves, is_square_attacked_static
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager as CacheManager

class TestMoveFiltering(unittest.TestCase):
    def setUp(self):
        # Create a real board and cache manager for realistic testing
        self.board = Board()
        self.cache_manager = CacheManager(self.board)
        self.board.cache_manager = self.cache_manager
        
        # Create a game state
        self.game_state = GameState(
            board=self.board,
            color=Color.WHITE,
            cache_manager=self.cache_manager,
            history=[],
            halfmove_clock=0,
            turn_number=1
        )

    def test_priest_exemption(self):
        """Test that self-check is ALLOWED when a priest is present."""
        # Setup: White King at (0,0,0), White Priest at (0,0,1)
        # Black Rook at (5,0,0) attacking (0,0,0)
        # Move: King moves to (1,0,0) - still attacked by Rook? No, Rook attacks rank.
        # Let's make it simpler.
        # White King at (0,0,0). Black Rook at (0,5,0).
        # King is in check.
        # Move: King to (1,0,0). Safe.
        # Move: King to (0,1,0). Still in check (on file).
        
        # But we want to test "legal moves that leave the current player's king in check".
        # So we need a move that results in check.
        
        # Scenario:
        # White King at (0,0,0).
        # White Pawn at (1,0,0).
        # Black Rook at (2,0,0).
        # If Pawn moves, King is exposed to check.
        
        # Clear board
        self.cache_manager.occupancy_cache.clear()
        
        # Place pieces
        self.cache_manager.occupancy_cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
        self.cache_manager.occupancy_cache.set_position(np.array([1,0,0]), np.array([PieceType.PAWN, Color.WHITE]))
        self.cache_manager.occupancy_cache.set_position(np.array([5,0,0]), np.array([PieceType.ROOK, Color.BLACK]))
        
        # Add a Priest
        self.cache_manager.occupancy_cache.set_position(np.array([0,0,1]), np.array([PieceType.PRIEST, Color.WHITE]))
        
        # Generate moves for Pawn at (1,0,0)
        # Pawn moves to (2,0,0) (capture) or (1,1,0) etc.
        # Let's say Pawn moves to (1,1,0).
        # King at (0,0,0) is now exposed to Rook at (5,0,0)? No, Rook is at (5,0,0).
        # Wait, (0,0,0) and (5,0,0) are on same x-axis? No, same y and z.
        # x varies. (0,0,0) -> (1,0,0) -> ... -> (5,0,0).
        # So Pawn at (1,0,0) BLOCKS the Rook.
        # If Pawn moves to (1,1,0), King is exposed.
        
        # Create the unsafe move
        unsafe_move = np.array([(1,0,0, 1,1,0, False, 0)], dtype=MOVE_DTYPE)
        
        # Filter moves
        # Since Priest exists, unsafe move should be KEPT.
        filtered = filter_safe_moves(self.game_state, unsafe_move)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['from_x'], 1)
        self.assertEqual(filtered[0]['to_y'], 1)

    def test_no_priest_filtering(self):
        """Test that self-check is PREVENTED when NO priest is present."""
        # Same scenario as above, but NO Priest.
        
        # Clear board
        self.cache_manager.occupancy_cache.clear()
        
        # Place pieces
        self.cache_manager.occupancy_cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
        self.cache_manager.occupancy_cache.set_position(np.array([1,0,0]), np.array([PieceType.PAWN, Color.WHITE]))
        self.cache_manager.occupancy_cache.set_position(np.array([5,0,0]), np.array([PieceType.ROOK, Color.BLACK]))
        
        # NO Priest
        
        # Create the unsafe move (Pawn moves out of the way)
        unsafe_move = np.array([(1,0,0, 1,1,0, False, 0)], dtype=MOVE_DTYPE)
        
        # Filter moves
        # Since NO Priest, unsafe move should be REMOVED.
        filtered = filter_safe_moves(self.game_state, unsafe_move)
        
        self.assertEqual(len(filtered), 0)

    def test_king_move_into_check(self):
        """Test King moving directly into check."""
        # Clear board
        self.cache_manager.occupancy_cache.clear()
        
        # Place pieces
        self.cache_manager.occupancy_cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
        self.cache_manager.occupancy_cache.set_position(np.array([2,0,0]), np.array([PieceType.ROOK, Color.BLACK]))
        
        # King moves to (1,0,0) -> adjacent to Rook? No, Rook attacks (1,0,0).
        # Move King to (1,0,0).
        
        unsafe_move = np.array([(0,0,0, 1,0,0, False, 0)], dtype=MOVE_DTYPE)
        
        # Filter moves
        filtered = filter_safe_moves(self.game_state, unsafe_move)
        
        self.assertEqual(len(filtered), 0)
        
    def test_safe_move(self):
        """Test a move that is safe is kept."""
        # Clear board
        self.cache_manager.occupancy_cache.clear()
        
        # Place pieces
        self.cache_manager.occupancy_cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
        # No enemies
        
        safe_move = np.array([(0,0,0, 1,0,0, False, 0)], dtype=MOVE_DTYPE)
        
        # Filter moves
        filtered = filter_safe_moves(self.game_state, safe_move)
        
        self.assertEqual(len(filtered), 1)

    def test_archer_attack_static(self):
        """Test static attack detection for Archer (distance 2)."""
        # Clear board
        self.cache_manager.occupancy_cache.clear()
        
        # Place Archer at (0,0,0)
        self.cache_manager.occupancy_cache.set_position(np.array([0,0,0]), np.array([PieceType.ARCHER, Color.BLACK]))
        
        # Check square at (0,0,2) -> Distance 2 -> Attacked
        is_attacked = is_square_attacked_static(self.game_state, np.array([0,0,2]), Color.BLACK)
        self.assertTrue(is_attacked)
        
        # Check square at (0,0,1) -> Distance 1 -> Not Attacked (Archer only attacks at dist 2)
        is_attacked = is_square_attacked_static(self.game_state, np.array([0,0,1]), Color.BLACK)
        self.assertFalse(is_attacked)

if __name__ == '__main__':
    unittest.main()
