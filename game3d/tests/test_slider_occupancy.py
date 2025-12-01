
import numpy as np
import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.common.shared_types import PieceType, Color, COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch

class TestSliderOccupancy(unittest.TestCase):
    def setUp(self):
        # Setup empty board
        self.board = Board.empty()
        empty_coords = np.empty((0, 3), dtype=COORD_DTYPE)
        empty_types = np.empty(0, dtype=PIECE_TYPE_DTYPE)
        empty_colors = np.empty(0, dtype=COLOR_DTYPE)
        
        self.cache = OptimizedCacheManager(self.board, Color.WHITE, initial_data=(empty_coords, empty_types, empty_colors))
        self.state = GameState(board=self.board, color=Color.WHITE, cache_manager=self.cache)
        
        # Ensure generator is initialized
        from game3d.movement import generator
        generator.initialize_generator()

    def test_spiral_occupancy(self):
        """Verify Spiral piece respects occupancy for pseudolegal moves and ignores it for raw moves."""
        # Place White Spiral at center (4, 4, 4)
        center = np.array([4, 4, 4], dtype=COORD_DTYPE)
        self.cache.occupancy_cache.set_position(center, (PieceType.SPIRAL, Color.WHITE))
        
        # Place Black Pawn blocking one ray (e.g. +X direction)
        # Spiral +X ray: [1, 0, 0]
        blocker_pos = np.array([6, 4, 4], dtype=COORD_DTYPE) # 2 steps away
        self.cache.occupancy_cache.set_position(blocker_pos, (PieceType.PAWN, Color.BLACK))
        
        # Place White Pawn blocking another ray (e.g. +Y direction)
        # Spiral +Y ray: [0, 1, 0]
        friendly_pos = np.array([4, 6, 4], dtype=COORD_DTYPE) # 2 steps away
        self.cache.occupancy_cache.set_position(friendly_pos, (PieceType.PAWN, Color.WHITE))
        
        # 1. Pseudolegal Moves (ignore_occupancy=False)
        moves_blocked = generate_pseudolegal_moves_batch(
            self.state, 
            np.array([center]), 
            ignore_occupancy=False
        )
        
        # Check enemy blocker
        has_capture = np.any(np.all(moves_blocked[:, 3:] == blocker_pos, axis=1))
        has_past_blocker = np.any(np.all(moves_blocked[:, 3:] == np.array([7, 4, 4]), axis=1))
        
        self.assertTrue(has_capture, "Should capture enemy blocker")
        self.assertFalse(has_past_blocker, "Should NOT move past enemy blocker")
        
        # Check friendly blocker
        has_friendly = np.any(np.all(moves_blocked[:, 3:] == friendly_pos, axis=1))
        has_past_friendly = np.any(np.all(moves_blocked[:, 3:] == np.array([4, 7, 4]), axis=1))
        
        self.assertFalse(has_friendly, "Should NOT capture friendly piece")
        self.assertFalse(has_past_friendly, "Should NOT move past friendly piece")
        
        # 2. Raw Moves (ignore_occupancy=True)
        moves_raw = generate_pseudolegal_moves_batch(
            self.state, 
            np.array([center]), 
            ignore_occupancy=True
        )
        
        # Check enemy blocker
        has_past_blocker_raw = np.any(np.all(moves_raw[:, 3:] == np.array([7, 4, 4]), axis=1))
        self.assertTrue(has_past_blocker_raw, "Raw moves SHOULD go through enemy piece")
        
        # Check friendly blocker
        has_past_friendly_raw = np.any(np.all(moves_raw[:, 3:] == np.array([4, 7, 4]), axis=1))
        self.assertTrue(has_past_friendly_raw, "Raw moves SHOULD go through friendly piece")

if __name__ == "__main__":
    unittest.main()
