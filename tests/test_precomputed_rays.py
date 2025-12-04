
import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from game3d.movement.slider_engine import get_slider_movement_generator
from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, Color, PieceType
from game3d.pieces.pieces.queen import QUEEN_MOVEMENT_VECTORS
from game3d.pieces.pieces.vectorslider import VECTOR_DIRECTIONS
from game3d.pieces.pieces.reflector import generate_reflecting_bishop_moves

class TestPrecomputedRays(unittest.TestCase):
    def setUp(self):
        self.engine = get_slider_movement_generator()
        self.cache_manager = MagicMock()
        # Mock occupancy: empty board
        self.occ = np.zeros((SIZE, SIZE, SIZE), dtype=np.int32)
        self.cache_manager.occupancy_cache._occ = self.occ
        self.color = Color.WHITE

    def test_queen_moves(self):
        print("Testing QUEEN moves...")
        pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
        
        # Legacy
        legacy_moves = self.engine.generate_slider_moves_array(
            self.cache_manager, self.color, pos, QUEEN_MOVEMENT_VECTORS, 8
        )
        
        # Precomputed
        precomputed_moves = self.engine.generate_slider_moves_precomputed(
            self.cache_manager, self.color, pos, "QUEEN"
        )
        
        # Compare sets of destinations (cols 3,4,5)
        legacy_dests = set(map(tuple, legacy_moves[:, 3:6]))
        precomputed_dests = set(map(tuple, precomputed_moves[:, 3:6]))
        
        self.assertEqual(legacy_dests, precomputed_dests)
        print("QUEEN moves match.")

    def test_vectorslider_moves(self):
        print("Testing VECTORSLIDER moves...")
        pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
        
        legacy_moves = self.engine.generate_slider_moves_array(
            self.cache_manager, self.color, pos, VECTOR_DIRECTIONS, 8
        )
        
        precomputed_moves = self.engine.generate_slider_moves_precomputed(
            self.cache_manager, self.color, pos, "VECTORSLIDER"
        )
        
        legacy_dests = set(map(tuple, legacy_moves[:, 3:6]))
        precomputed_dests = set(map(tuple, precomputed_moves[:, 3:6]))
        
        self.assertEqual(legacy_dests, precomputed_dests)
        print("VECTORSLIDER moves match.")

    def test_reflector_moves(self):
        print("Testing REFLECTOR moves...")
        # Reflector logic is custom in legacy, so we compare against generate_reflecting_bishop_moves
        pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
        
        # Mock flattened occupancy for legacy reflector
        self.cache_manager.occupancy_cache.get_flattened_occupancy.return_value = self.occ.ravel()
        
        legacy_moves = generate_reflecting_bishop_moves(
            self.cache_manager, self.color, pos, max_bounces=2
        )
        
        precomputed_moves = self.engine.generate_slider_moves_precomputed(
            self.cache_manager, self.color, pos, "REFLECTOR"
        )
        
        legacy_dests = set(map(tuple, legacy_moves[:, 3:6]))
        precomputed_dests = set(map(tuple, precomputed_moves[:, 3:6]))
        
        self.assertEqual(legacy_dests, precomputed_dests)
        print("REFLECTOR moves match.")

if __name__ == '__main__':
    unittest.main()
