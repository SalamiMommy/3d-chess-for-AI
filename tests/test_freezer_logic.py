import unittest
import numpy as np
import sys
import os

# Add the project root to the path so we can import game3d
sys.path.append(os.getcwd())

from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.pieces.pieces.freezer import FREEZER_MOVEMENT_VECTORS
from game3d.cache.effectscache.auracache import EFFECT_FREEZE, EFFECT_BUFF, EFFECT_DEBUFF, EFFECT_PULL, EFFECT_PUSH

class TestFreezerLogic(unittest.TestCase):
    def setUp(self):
        from game3d.board.board import Board
        self.board = Board.empty()
        self.state = GameState(self.board)


    def test_freezer_frequency(self):
        """Test that freezers only freeze every 2nd iteration of friendly turns."""
        # Setup: White Freezer at (3, 3, 3), Black Pawn at (3, 3, 4) (adjacent)
        freezer_pos = np.array([3, 3, 3], dtype=COORD_DTYPE)
        enemy_pos = np.array([3, 3, 4], dtype=COORD_DTYPE)
        
        self.state.cache_manager.occupancy_cache.set_position(
            freezer_pos, 
            np.array([PieceType.FREEZER, Color.WHITE])
        )
        self.state.cache_manager.occupancy_cache.set_position(
            enemy_pos, 
            np.array([PieceType.PAWN, Color.BLACK])
        )
        
        # Manually trigger freeze for Turn 0 (White's 1st turn)
        # Should FREEZE
        self.state.cache_manager.consolidated_aura_cache.trigger_freeze(Color.WHITE, 0)
        
        # Check if enemy is frozen
        is_frozen = self.state.cache_manager.consolidated_aura_cache.batch_is_frozen(
            enemy_pos.reshape(1, 3), 0, Color.BLACK
        )[0]
        self.assertTrue(is_frozen, "Enemy should be frozen on Turn 0 (White's 1st turn)")
        
        # Turn 1 (Black's turn) - Enemy is frozen, so they can't move (conceptually)
        # But we are testing the APPLICATION of freeze for the NEXT turn.
        # The freeze applied at Turn 0 expires at Turn 1.
        
        # Turn 2 (White's 2nd turn)
        # Should NOT FREEZE (every 2nd iteration logic: Active, Inactive, Active...)
        
        self.state.cache_manager.consolidated_aura_cache.trigger_freeze(Color.WHITE, 2)
        
        # Check if enemy is frozen for Turn 2 (would affect Turn 3)
        # We need to check if the freeze expiry was updated.
        # If it skipped, expiry should still be old (from Turn 0, which was 1).
        # Current turn is 2.
        is_frozen = self.state.cache_manager.consolidated_aura_cache.batch_is_frozen(
            enemy_pos.reshape(1, 3), 2, Color.BLACK
        )[0]
        
        # If logic is implemented, this should be FALSE because we skipped applying it.
        # BUT, since we haven't implemented it yet, this test will FAIL (it will be True).
        self.assertFalse(is_frozen, "Enemy should NOT be frozen on Turn 2 (White's 2nd turn)")
        
        # Turn 4 (White's 3rd turn)
        # Should FREEZE
        self.state.cache_manager.consolidated_aura_cache.trigger_freeze(Color.WHITE, 4)
        is_frozen = self.state.cache_manager.consolidated_aura_cache.batch_is_frozen(
            enemy_pos.reshape(1, 3), 4, Color.BLACK
        )[0]
        self.assertTrue(is_frozen, "Enemy should be frozen on Turn 4 (White's 3rd turn)")

    def test_incremental_aura_updates(self):
        """Verify that moving an aura piece updates the aura cache."""
        # Setup: White Speeder at (2, 2, 2)
        dest_pos = np.array([2, 2, 3], dtype=COORD_DTYPE)
        
        # Place friendly piece to receive buff
        nearby = np.array([2, 2, 1], dtype=COORD_DTYPE).reshape(1, 3)
        self.state.cache_manager.occupancy_cache.set_position(
            nearby, 
            np.array([PieceType.PAWN, Color.WHITE])
        )

        # Place speeder
        speeder_pos = np.array([2, 2, 2], dtype=COORD_DTYPE).reshape(1, 3)
        self.state.cache_manager.occupancy_cache.set_position(
            speeder_pos, 
            np.array([PieceType.SPEEDER, Color.WHITE])
        )
        
        # Manually notify aura cache for initial setup (Speeder)
        self.state.cache_manager._notify_all_effect_caches(
            speeder_pos, 
            np.array([[PieceType.SPEEDER, Color.WHITE]], dtype=int)
        )
        
        # Verify initial buff
        is_buffed = self.state.cache_manager.consolidated_aura_cache.batch_is_buffed(nearby, Color.WHITE)[0]
        self.assertTrue(is_buffed, "Square should be buffed initially")
        
        # Move speeder to (2, 2, 5) - far enough to clear buff at (2, 2, 1)
        dest_pos = np.array([2, 2, 5], dtype=COORD_DTYPE)
        
        # Move Speeder
        from game3d.game.turnmove import make_move
        # Move speeder to (2, 2, 5)
        move = np.concatenate([speeder_pos.flatten(), dest_pos])
        
        # We need to mock legal moves or bypass validation for this unit test to run fast
        # But make_move does validation.
        # Instead, we just simulate the cache updates as turnmove would do.
        # turnmove calls: cache_manager._notify_all_effect_caches
        
        # Simulate the move update in cache
        changed_coords = np.array([speeder_pos.flatten(), dest_pos], dtype=COORD_DTYPE)
        pieces_data = np.array([
            [0, 0],  # Source empty
            [PieceType.SPEEDER, Color.WHITE]  # Dest occupied
        ], dtype=int)
        
        # Place new friendly piece to receive buff
        new_nearby = np.array([2, 2, 6], dtype=COORD_DTYPE).reshape(1, 3)
        self.state.cache_manager.occupancy_cache.set_position(
            new_nearby, 
            np.array([PieceType.PAWN, Color.WHITE])
        )

        # Update occupancy first (as turnmove does)
        self.state.cache_manager.occupancy_cache.batch_set_positions(changed_coords, pieces_data)
        
        # Notify effect caches
        self.state.cache_manager._notify_all_effect_caches(changed_coords, pieces_data)
        
        # Verify old buff is gone
        is_buffed_old = self.state.cache_manager.consolidated_aura_cache.batch_is_buffed(nearby, Color.WHITE)[0]
        self.assertFalse(is_buffed_old, "Old square should no longer be buffed")
        
        # Verify new buff exists
        is_buffed_new = self.state.cache_manager.consolidated_aura_cache.batch_is_buffed(new_nearby, Color.WHITE)[0]
        self.assertTrue(is_buffed_new, "New square should be buffed")

if __name__ == '__main__':
    unittest.main()
