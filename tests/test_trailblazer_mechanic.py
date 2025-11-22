import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE
from game3d.cache.manager import OptimizedCacheManager
from game3d.game.turnmove import make_move

class TestTrailblazerMechanic(unittest.TestCase):
    def setUp(self):
        self.board = Board.empty()
        self.cache_manager = OptimizedCacheManager(self.board)
        self.game_state = GameState(self.board, Color.WHITE, self.cache_manager)
        
        # Clear caches explicitly just in case
        self.cache_manager.occupancy_cache.clear()
        self.cache_manager.trailblaze_cache.clear()

    def test_trail_creation_and_counters(self):
        # 1. Setup
        # White Trailblazer at (0,0,0)
        tb_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        self.board.set_piece_at(tb_pos, PieceType.TRAILBLAZER, Color.WHITE)
        self.cache_manager.occupancy_cache.set_position(tb_pos, (PieceType.TRAILBLAZER, Color.WHITE))
        
        # Black Rook at (0,0,5)
        rook_pos = np.array([0, 0, 5], dtype=COORD_DTYPE)
        self.board.set_piece_at(rook_pos, PieceType.ROOK, Color.BLACK)
        self.cache_manager.occupancy_cache.set_position(rook_pos, (PieceType.ROOK, Color.BLACK))
        
        # 2. Move Trailblazer (0,0,0) -> (0,0,3)
        # Path: (0,0,1), (0,0,2)
        move_tb = np.array([0, 0, 0, 0, 0, 3], dtype=COORD_DTYPE)
        self.game_state = make_move(self.game_state, move_tb)
        
        # Verify trails
        trails = self.cache_manager.trailblaze_cache.get_all_trails()
        # Should have trails at (0,0,1) and (0,0,2)
        # Note: get_all_trails returns dict of {flat_idx: coords}
        # We can check intersecting squares
        check_path = np.array([[0, 0, 1], [0, 0, 2]], dtype=COORD_DTYPE)
        self.assertTrue(self.cache_manager.trailblaze_cache.check_trail_intersection(check_path))
        
        # 3. Move Black Rook (0,0,5) -> (0,0,4)
        # Path: None (adjacent)
        # Landing: (0,0,4) (Not on trail)
        move_rook_1 = np.array([0, 0, 5, 0, 0, 4], dtype=COORD_DTYPE)
        self.game_state = make_move(self.game_state, move_rook_1)
        
        # Verify counters = 0
        rook_pos_new = np.array([0, 0, 4], dtype=COORD_DTYPE)
        counters = self.cache_manager.trailblaze_cache.get_counter(rook_pos_new)
        self.assertEqual(counters, 0)
        
        # 4. Move Trailblazer (0,0,3) -> (0,0,0)
        # Path: (0,0,2), (0,0,1) (Adds to history)
        move_tb_back = np.array([0, 0, 3, 0, 0, 0], dtype=COORD_DTYPE)
        self.game_state = make_move(self.game_state, move_tb_back)
        
        # 5. Move Black Rook (0,0,4) -> (0,0,1)
        # Path: (0,0,3), (0,0,2)
        # Landing: (0,0,1)
        # Trails are at (0,0,1) and (0,0,2) from previous moves.
        # Intersection: (0,0,2) is on trail. (0,0,3) is NOT on trail.
        # Landing: (0,0,1) is on trail.
        # Total counters expected: 1 (path) + 1 (landing) = 2.
        move_rook_2 = np.array([0, 0, 4, 0, 0, 1], dtype=COORD_DTYPE)
        self.game_state = make_move(self.game_state, move_rook_2)
        
        # Verify counters
        rook_pos_final = np.array([0, 0, 1], dtype=COORD_DTYPE)
        counters = self.cache_manager.trailblaze_cache.get_counter(rook_pos_final)
        self.assertEqual(counters, 2)
        
        # 6. Move Rook again to trigger capture
        # Move (0,0,1) -> (0,0,2)
        # Path: None
        # Landing: (0,0,2) (On trail)
        # Counter +1 -> 3 -> Capture
        move_rook_3 = np.array([0, 0, 1, 0, 0, 2], dtype=COORD_DTYPE)
        self.game_state = make_move(self.game_state, move_rook_3)
        
        # Verify capture
        piece_at_dest = self.game_state.board.get_piece_at(np.array([0, 0, 2], dtype=COORD_DTYPE))
        self.assertIsNone(piece_at_dest) # Should be empty/None
        
        # Verify counters cleared
        counters = self.cache_manager.trailblaze_cache.get_counter(np.array([0, 0, 2], dtype=COORD_DTYPE))
        self.assertEqual(counters, 0)

if __name__ == '__main__':
    unittest.main()
