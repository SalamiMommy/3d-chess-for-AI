import unittest
import numpy as np
from game3d.game.gamestate import GameState
import game3d.movement.generator
print(f"DEBUG: generator file: {game3d.movement.generator.__file__}")
from game3d.movement.generator import generate_legal_moves
from game3d.game.turnmove import legal_moves_for_piece
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE, SIZE
from game3d.pieces.pieces.wall import is_wall_anchor

class TestWallMechanics(unittest.TestCase):
    def setUp(self):
        self.state = GameState.from_startpos()
        self.cache_manager = self.state.cache_manager
        self.occ_cache = self.cache_manager.occupancy_cache

    def test_wall_spawning(self):
        """Verify Walls spawn as 2x2 blocks."""
        # Check White Wall at (1,1)
        # Should occupy (1,1), (2,1), (1,2), (2,2)
        # Note: Board layout uses (row, col) which maps to (y, x) or (x, y)?
        # board.py: coords[..., 0] = valid_x, coords[..., 1] = valid_y
        # rank2_layout is [y, x] if numpy convention.
        # Row 1 (index 1) is y=1. Col 1 (index 1) is x=1.
        # So (1,1) is Wall.
        # We filled (1,2), (2,1), (2,2) with Wall.
        
        # Check (1,1,1) - White Wall (Z=1)
        # Wait, rank 2 is Z=1 for White.
        
        expected_squares = [
            (1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)
        ]
        
        for x, y, z in expected_squares:
            coord = np.array([x, y, z], dtype=COORD_DTYPE)
            ptype, color = self.occ_cache.get_fast(coord)
            self.assertEqual(ptype, PieceType.WALL, f"Square {coord} should be Wall")
            self.assertEqual(color, Color.WHITE, f"Square {coord} should be White")

    def test_wall_anchor_logic(self):
        """Verify only top-left generates moves."""
        # Anchor should be (1,1,1)
        anchor = np.array([1, 1, 1], dtype=COORD_DTYPE)
        self.assertTrue(is_wall_anchor(anchor, self.cache_manager))
        
        # Others should not be anchors
        others = [
            (2, 1, 1), (1, 2, 1), (2, 2, 1)
        ]
        for x, y, z in others:
            coord = np.array([x, y, z], dtype=COORD_DTYPE)
            self.assertFalse(is_wall_anchor(coord, self.cache_manager), f"{coord} should not be anchor")

    def test_wall_movement(self):
        """Verify 2x2 block movement."""
        # Clear board to isolate wall
        self.occ_cache.clear()
        
        # Place Kings (required for move generation)
        self.occ_cache.set_position_fast(np.array([0, 0, 0], dtype=COORD_DTYPE), PieceType.KING, Color.WHITE)
        self.occ_cache.set_position_fast(np.array([8, 8, 8], dtype=COORD_DTYPE), PieceType.KING, Color.BLACK)
        
        # Place White Wall at (3,3,3)
        # Anchor (3,3,3). Block: (3,3), (4,3), (3,4), (4,4)
        wall_squares = [
            (3, 3, 3), (4, 3, 3), (3, 4, 3), (4, 4, 3)
        ]
        for x, y, z in wall_squares:
            self.occ_cache.set_position_fast(np.array([x, y, z], dtype=COORD_DTYPE), PieceType.WALL, Color.WHITE)
            
        # Generate moves for anchor
        anchor = np.array([3, 3, 3], dtype=COORD_DTYPE)
        
        moves = legal_moves_for_piece(self.state, anchor)
        
        # Should be able to move in 6 directions if empty
        # +X: (4,3,3) -> (5,3,3). Block moves to (4,3), (5,3), (4,4), (5,4).
        # Check if +X move exists
        # from (3,3,3) to (4,3,3)
        has_move_px = np.any(np.all(moves[:, 3:] == np.array([4, 3, 3]), axis=1))
        self.assertTrue(has_move_px, "Should move +X")
        
        # Execute move
        move = np.array([3, 3, 3, 4, 3, 3], dtype=COORD_DTYPE)
        self.state.make_move_vectorized(move)
        
        # Verify new position
        # Old squares should be empty (except overlap)
        # New block: (4,3), (5,3), (4,4), (5,4)
        # Overlap: (4,3) and (4,4) were part of old wall, now part of new wall.
        # (3,3) and (3,4) should be empty.
        
        self.assertEqual(self.occ_cache.get_fast(np.array([3, 3, 3], dtype=COORD_DTYPE))[0], 0, "(3,3) should be empty")
        self.assertEqual(self.occ_cache.get_fast(np.array([3, 4, 3], dtype=COORD_DTYPE))[0], 0, "(3,4) should be empty")
        
        new_squares = [
            (4, 3, 3), (5, 3, 3), (4, 4, 3), (5, 4, 3)
        ]
        for x, y, z in new_squares:
            ptype, color = self.occ_cache.get_fast(np.array([x, y, z], dtype=COORD_DTYPE))
            self.assertEqual(ptype, PieceType.WALL, f"New square {x,y,z} should be Wall")

    def test_capture_filtering(self):
        """Verify capture filtering (in front vs behind)."""
        self.occ_cache.clear()
        
        # Place Kings
        self.occ_cache.set_position(np.array([0, 0, 0], dtype=COORD_DTYPE), np.array([PieceType.KING, Color.WHITE]))
        self.occ_cache.set_position(np.array([8, 8, 8], dtype=COORD_DTYPE), np.array([PieceType.KING, Color.BLACK]))
        
        # Place White Wall at Z=2
        # Anchor (3,3,2)
        wall_squares = [(3, 3, 2), (4, 3, 2), (3, 4, 2), (4, 4, 2)]
        for x, y, z in wall_squares:
            self.occ_cache.set_position(np.array([x, y, z], dtype=COORD_DTYPE), np.array([PieceType.WALL, Color.WHITE]))
            
        # Place Black Rook in front (Z=3)
        # At (3,3,3) - directly above anchor
        attacker_pos = np.array([3, 3, 3], dtype=COORD_DTYPE)
        self.occ_cache.set_position(attacker_pos, np.array([PieceType.ROOK, Color.BLACK]))
        
        # Set turn to Black
        self.state.color = Color.BLACK
        
        # Generate moves for Rook
        moves = legal_moves_for_piece(self.state, attacker_pos)
        
        # Should NOT be able to capture (3,3,2) because 3 > 2 (in front of White Wall)
        can_capture = np.any(np.all(moves[:, 3:] == np.array([3, 3, 2]), axis=1))
        self.assertFalse(can_capture, "Should NOT capture White Wall from front (Z=3 > Z=2)")
        
        # Place Black Rook behind (Z=1)
        attacker_pos_behind = np.array([3, 3, 1], dtype=COORD_DTYPE)
        self.occ_cache.set_position(attacker_pos_behind, np.array([PieceType.ROOK, Color.BLACK]))
        
        # INVALIDATE CACHE to force regeneration
        self.state.cache_manager.move_cache.invalidate()
        
        moves_behind = legal_moves_for_piece(self.state, attacker_pos_behind)
        
        # Should be able to capture (3,3,2) because 1 < 2 (behind White Wall)
        can_capture_behind = np.any(np.all(moves_behind[:, 3:] == np.array([3, 3, 2]), axis=1))
        self.assertTrue(can_capture_behind, "Should capture White Wall from behind (Z=1 < Z=2)")

if __name__ == '__main__':
    unittest.main()
