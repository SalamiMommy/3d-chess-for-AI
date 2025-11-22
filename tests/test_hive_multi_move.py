"""Tests for multi-hive move system."""

import unittest
import numpy as np
from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.game3d import OptimizedGame3D
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE
from game3d.movement.movepiece import Move


class TestMultiHiveMove(unittest.TestCase):
    """Test suite for multi-hive move functionality."""
    
    def setUp(self):
        """Set up test game with multiple hives."""
        self.board = Board.empty()
        self.cache_manager = OptimizedCacheManager(self.board, Color.WHITE)
        self.game = OptimizedGame3D(board=self.board, cache=self.cache_manager)
        
        # Place 3 white hives at different positions
        self.hive1_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        self.hive2_pos = np.array([2, 2, 2], dtype=COORD_DTYPE)
        self.hive3_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
        
        self.board.set_piece_at(self.hive1_pos, PieceType.HIVE, Color.WHITE)
        self.board.set_piece_at(self.hive2_pos, PieceType.HIVE, Color.WHITE)
        self.board.set_piece_at(self.hive3_pos, PieceType.HIVE, Color.WHITE)
        
        # Update cache
        self.cache_manager.occupancy_cache.set_position(
            self.hive1_pos, np.array([PieceType.HIVE, Color.WHITE], dtype=np.int32)
        )
        self.cache_manager.occupancy_cache.set_position(
            self.hive2_pos, np.array([PieceType.HIVE, Color.WHITE], dtype=np.int32)
        )
        self.cache_manager.occupancy_cache.set_position(
            self.hive3_pos, np.array([PieceType.HIVE, Color.WHITE], dtype=np.int32)
        )
        
    def test_single_hive_move(self):
        """Test that a single hive move works as before."""
        # Move hive1 one step
        move = Move(self.hive1_pos, self.hive1_pos + np.array([1, 0, 0]))
        
        receipt = self.game.submit_move(move)
        
        # Should be valid
        self.assertTrue(receipt.is_legal)
        # Turn should switch (no other hives can move from their positions)
        # Actually, the other hives CAN move, so turn should NOT switch
        # Let me check the logic...
        # Since there are still unmoved hives (hive2 and hive3), turn should NOT switch
        self.assertEqual(self.game.current_player, Color.WHITE)
        
    def test_multiple_hive_moves(self):
        """Test moving multiple hives in sequence."""
        # Move hive1
        move1 = Move(self.hive1_pos, self.hive1_pos + np.array([1, 0, 0]))
        receipt1 = self.game.submit_move(move1)
        self.assertTrue(receipt1.is_legal)
        self.assertEqual(self.game.current_player, Color.WHITE)  # Turn shouldn't switch yet
        
        # Move hive2
        move2 = Move(self.hive2_pos, self.hive2_pos + np.array([0, 1, 0]))
        receipt2 = self.game.submit_move(move2)
        self.assertTrue(receipt2.is_legal)
        self.assertEqual(self.game.current_player, Color.WHITE)  # Still white's turn
        
        # Move hive3 - last hive
        move3 = Move(self.hive3_pos, self.hive3_pos + np.array([0, 0, 1]))
        receipt3 = self.game.submit_move(move3)
        self.assertTrue(receipt3.is_legal)
        # Now turn should switch since all hives have moved
        self.assertEqual(self.game.current_player, Color.BLACK)
        
    def test_duplicate_hive_move_rejected(self):
        """Test that moving the same hive twice is rejected."""
        # Move hive1 first time
        move1 = Move(self.hive1_pos, self.hive1_pos + np.array([1, 0, 0]))
        receipt1 = self.game.submit_move(move1)
        self.assertTrue(receipt1.is_legal)
        
        # Try to move hive1 again (from new position)
        new_pos = self.hive1_pos + np.array([1, 0, 0])
        move2 = Move(new_pos, new_pos + np.array([1, 0, 0]))
        
        from game3d.game3d import InvalidMoveError
        with self.assertRaises(InvalidMoveError) as context:
            self.game.submit_move(move2)
        
        self.assertIn("already moved", str(context.exception).lower())
        
    def test_hive_tracking_cleared_after_turn(self):
        """Test that hive move tracking is cleared after turn ends."""
        # Move all 3 hives
        move1 = Move(self.hive1_pos, self.hive1_pos + np.array([1, 0, 0]))
        move2 = Move(self.hive2_pos, self.hive2_pos + np.array([0, 1, 0]))
        move3 = Move(self.hive3_pos, self.hive3_pos + np.array([0, 0, 1]))
        
        self.game.submit_move(move1)
        self.game.submit_move(move2)
        self.game.submit_move(move3)
        
        # Turn should have switched to BLACK
        self.assertEqual(self.game.current_player, Color.BLACK)
        
        # Check that tracking is cleared
        self.assertEqual(len(self.game.state._moved_hive_positions), 0)
        self.assertEqual(len(self.game.state._pending_hive_moves), 0)
        
    def test_two_hive_scenario(self):
        """Test with only 2 hives."""
        # Remove hive3
        self.board.set_piece_at(self.hive3_pos, 0, Color.EMPTY)
        self.cache_manager.occupancy_cache.set_position(self.hive3_pos, None)
        
        # Move hive1
        move1 = Move(self.hive1_pos, self.hive1_pos + np.array([1, 0, 0]))
        self.game.submit_move(move1)
        self.assertEqual(self.game.current_player, Color.WHITE)
        
        # Move hive2 - should auto-finalize
        move2 = Move(self.hive2_pos, self.hive2_pos + np.array([0, 1, 0]))
        self.game.submit_move(move2)
        self.assertEqual(self.game.current_player, Color.BLACK)


if __name__ == '__main__':
    unittest.main()
