
import unittest
import numpy as np
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE
from game3d.core.structures import StructureManager

class TestAtomicStructures(unittest.TestCase):
    def setUp(self):
        # Create a blank board
        self.state = GameState(Board.empty(), Color.WHITE)
        self.cache = self.state.cache_manager
        
    def test_wall_creation_and_recognition(self):
        """Test that we can place a wall and StructureManager recognizes it."""
        # Place a 2x2 Wall at (4,4,4)
        # Wall anchor at (4,4,4) covers (4,4,4), (5,4,4), (4,5,4), (5,5,4)
        anchor = np.array([4, 4, 4], dtype=COORD_DTYPE)
        squares = StructureManager.get_full_structure_squares(anchor, PieceType.WALL)
        
        # Verify squares
        expected = {
            (4, 4, 4), (5, 4, 4), (4, 5, 4), (5, 5, 4)
        }
        computed = set((int(s[0]), int(s[1]), int(s[2])) for s in squares)
        self.assertEqual(expected, computed)
        
        # Place on board
        pieces_data = np.tile(np.array([PieceType.WALL, Color.WHITE], dtype=np.int8), (4, 1))
        self.cache.occupancy_cache.batch_set_positions(squares, pieces_data)
        
        # Verify anchor detection
        self.assertTrue(StructureManager.is_wall_anchor(anchor, self.cache.occupancy_cache))
        
        # Verify component detection
        for s in squares:
            found_anchor = StructureManager.find_anchor_for_square(s, PieceType.WALL, self.cache.occupancy_cache)
            self.assertTrue(np.array_equal(found_anchor, anchor), f"Failed for {s}")

    def test_atomic_wall_capture(self):
        """Test that capturing one part of a wall removes the entire wall."""
        # 1. Setup Wall at (4,4,4)
        anchor = np.array([4, 4, 4], dtype=COORD_DTYPE)
        wall_squares = StructureManager.get_full_structure_squares(anchor, PieceType.WALL)
        pieces_data = np.tile(np.array([PieceType.WALL, Color.WHITE], dtype=np.int8), (4, 1))
        self.cache.occupancy_cache.batch_set_positions(wall_squares, pieces_data)
        
        # 2. Setup Capturing Piece (Black Rook) at (4,4,5) attacks (4,4,4)
        attacker_pos = np.array([4, 4, 5], dtype=COORD_DTYPE)
        self.cache.occupancy_cache.set_position(attacker_pos, np.array([PieceType.ROOK, Color.BLACK]))
        
        # 3. Execute Capture Move: (4,4,5) -> (4,4,4)
        # This hits the wall anchor directly
        move = np.array([4, 4, 5, 4, 4, 4], dtype=np.int8) # Simplified move representation
        
        # We need to call make_move from turnmove.py
        from game3d.game.turnmove import make_move_trusted
        
        # We use trusted move because we manually set up the state and legal move generation might be finicky on empty board
        new_state = make_move_trusted(self.state, move)
        
        # 4. Verify Results
        # - Attacker should be at (4,4,4)
        # - ALL other wall squares should be EMPTY
        
        # Check dest
        ptype, color = new_state.cache_manager.occupancy_cache.get_fast(np.array([4, 4, 4]))
        self.assertEqual(ptype, PieceType.ROOK)
        self.assertEqual(color, Color.BLACK)
        
        # Check other wall squares
        other_squares = [
            (5, 4, 4), (4, 5, 4), (5, 5, 4)
        ]
        for x, y, z in other_squares:
            ptype, _ = new_state.cache_manager.occupancy_cache.get_fast(np.array([x, y, z]))
            self.assertEqual(ptype, 0, f"Square ({x},{y},{z}) should be empty after wall capture")

    def test_atomic_wall_capture_non_anchor(self):
        """Test capturing a non-anchor part of the wall."""
        # 1. Setup Wall at (4,4,4)
        anchor = np.array([4, 4, 4], dtype=COORD_DTYPE)
        wall_squares = StructureManager.get_full_structure_squares(anchor, PieceType.WALL)
        pieces_data = np.tile(np.array([PieceType.WALL, Color.WHITE], dtype=np.int8), (4, 1))
        self.cache.occupancy_cache.batch_set_positions(wall_squares, pieces_data)
        
        # 2. Setup Attacker at (5,4,5) attacking (5,4,4)
        attacker_pos = np.array([5, 4, 5], dtype=COORD_DTYPE)
        self.cache.occupancy_cache.set_position(attacker_pos, np.array([PieceType.ROOK, Color.BLACK]))
        
        # 3. Execute Move: (5,4,5) -> (5,4,4)
        move = np.array([5, 4, 5, 5, 4, 4], dtype=np.int8)
        
        from game3d.game.turnmove import make_move_trusted
        new_state = make_move_trusted(self.state, move)
        
        # 4. Verify
        # Capture square occupied by attacker
        ptype, color = new_state.cache_manager.occupancy_cache.get_fast(np.array([5, 4, 4]))
        self.assertEqual(ptype, PieceType.ROOK)
        
        # Anchor (4,4,4) should be GONE (Empty)
        ptype, _ = new_state.cache_manager.occupancy_cache.get_fast(np.array([4, 4, 4]))
        self.assertEqual(ptype, 0, "Anchor should be removed when part is captured")

if __name__ == '__main__':
    unittest.main()
