import unittest
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.attacks.pin import get_pinned_pieces
from game3d.movement.generator import generate_legal_moves
from game3d.movement.pseudolegal import coord_to_key

class TestPinDetection(unittest.TestCase):
    def setUp(self):
        self.board = Board()
        self.state = GameState(self.board)
        # Clear occupancy explicitly just in case
        self.state.cache_manager.occupancy_cache.clear()

    def set_piece(self, x, y, z, piece_type, color):
        coord = np.array([x, y, z], dtype=COORD_DTYPE)
        piece = np.array([piece_type, color], dtype=np.int8) # Assuming int8 or similar for piece data
        self.state.cache_manager.occupancy_cache.set_position(coord, piece)

    def test_raw_vs_pseudolegal_moves(self):
        """Test that raw moves ignore occupancy and pseudolegal moves respect it."""
        print("\nTesting Raw vs Pseudolegal Moves...")
        # Setup: White Rook at (0,0,0), Black Pawn at (0,2,0)
        # Raw moves should go through pawn. Pseudolegal should stop at pawn.
        
        self.set_piece(0, 0, 0, PieceType.ROOK, Color.WHITE)
        self.set_piece(0, 2, 0, PieceType.PAWN, Color.BLACK)
        
        # Force cache update
        self.state.color = Color.WHITE
        generate_legal_moves(self.state)
        
        # Get moves from cache
        raw_moves = self.state.cache_manager.move_cache.get_raw_moves(Color.WHITE)
        pseudo_moves = self.state.cache_manager.move_cache.get_pseudolegal_moves(Color.WHITE)
        
        if raw_moves is None:
            self.fail("Raw moves cache is empty")
        if pseudo_moves is None:
            self.fail("Pseudolegal moves cache is empty")

        # Check raw moves
        # Should contain (0,3,0) which is behind the pawn
        raw_destinations = raw_moves[:, 3:]
        has_behind = np.any(np.all(raw_destinations == [0, 3, 0], axis=1))
        self.assertTrue(has_behind, "Raw moves should include square behind blocker")
        
        # Check pseudolegal moves
        # Should NOT contain (0,3,0)
        pseudo_destinations = pseudo_moves[:, 3:]
        has_behind_pseudo = np.any(np.all(pseudo_destinations == [0, 3, 0], axis=1))
        self.assertFalse(has_behind_pseudo, "Pseudolegal moves should NOT include square behind blocker")
        
        # Both should contain capture of pawn at (0,2,0)
        has_capture_raw = np.any(np.all(raw_destinations == [0, 2, 0], axis=1))
        has_capture_pseudo = np.any(np.all(pseudo_destinations == [0, 2, 0], axis=1))
        self.assertTrue(has_capture_raw, "Raw moves should include capture")
        self.assertTrue(has_capture_pseudo, "Pseudolegal moves should include capture")
        print("Raw vs Pseudolegal Moves Test Passed!")

    def test_pin_detection_orthogonal(self):
        """Test pin detection on a rank/file."""
        print("\nTesting Orthogonal Pin...")
        # Setup:
        # White King at (0,0,0)
        # White Pawn at (0,2,0) (Pinned)
        # Black Rook at (0,5,0) (Pinner)
        
        self.set_piece(0, 0, 0, PieceType.KING, Color.WHITE)
        self.set_piece(0, 2, 0, PieceType.PAWN, Color.WHITE)
        self.set_piece(0, 5, 0, PieceType.ROOK, Color.BLACK)
        
        # We need to generate moves for BLACK first to populate raw moves cache
        self.state.color = Color.BLACK
        generate_legal_moves(self.state)
        
        # Now check pins for WHITE
        pinned = get_pinned_pieces(self.state, Color.WHITE)
        
        pawn_key = int(coord_to_key(np.array([[0, 2, 0]], dtype=COORD_DTYPE))[0])
        rook_key = int(coord_to_key(np.array([[0, 5, 0]], dtype=COORD_DTYPE))[0])
        
        self.assertIn(pawn_key, pinned, "Pawn should be pinned")
        self.assertEqual(pinned[pawn_key], rook_key, "Pawn should be pinned by Rook")
        print("Orthogonal Pin Test Passed!")

    def test_pin_detection_diagonal(self):
        """Test pin detection on a diagonal."""
        print("\nTesting Diagonal Pin...")
        # Setup:
        # White King at (0,0,0)
        # White Bishop at (2,2,0) (Pinned)
        # Black Queen at (5,5,0) (Pinner)
        
        self.set_piece(0, 0, 0, PieceType.KING, Color.WHITE)
        self.set_piece(2, 2, 0, PieceType.BISHOP, Color.WHITE)
        self.set_piece(5, 5, 0, PieceType.QUEEN, Color.BLACK)
        
        # Generate moves for BLACK
        self.state.color = Color.BLACK
        generate_legal_moves(self.state)
        
        # Check pins for WHITE
        pinned = get_pinned_pieces(self.state, Color.WHITE)
        
        bishop_key = int(coord_to_key(np.array([[2, 2, 0]], dtype=COORD_DTYPE))[0])
        queen_key = int(coord_to_key(np.array([[5, 5, 0]], dtype=COORD_DTYPE))[0])
        
        self.assertIn(bishop_key, pinned, "Bishop should be pinned")
        self.assertEqual(pinned[bishop_key], queen_key, "Bishop should be pinned by Queen")
        print("Diagonal Pin Test Passed!")

    def test_no_pin_with_two_pieces(self):
        """Test that two pieces blocking means neither is pinned."""
        print("\nTesting Double Block (No Pin)...")
        # Setup:
        # White King at (0,0,0)
        # White Pawn at (0,2,0)
        # White Pawn at (0,3,0)
        # Black Rook at (0,5,0)
        
        self.set_piece(0, 0, 0, PieceType.KING, Color.WHITE)
        self.set_piece(0, 2, 0, PieceType.PAWN, Color.WHITE)
        self.set_piece(0, 3, 0, PieceType.PAWN, Color.WHITE)
        self.set_piece(0, 5, 0, PieceType.ROOK, Color.BLACK)
        
        # Generate moves for BLACK
        self.state.color = Color.BLACK
        generate_legal_moves(self.state)
        
        # Check pins for WHITE
        pinned = get_pinned_pieces(self.state, Color.WHITE)
        
        self.assertEqual(len(pinned), 0, "No pieces should be pinned (double block)")
        print("Double Block Test Passed!")

    def test_no_pin_with_enemy_blocker(self):
        """Test that an enemy piece blocking means no pin."""
        print("\nTesting Enemy Block (No Pin)...")
        # Setup:
        # White King at (0,0,0)
        # White Pawn at (0,2,0)
        # Black Pawn at (0,3,0)
        # Black Rook at (0,5,0)
        
        self.set_piece(0, 0, 0, PieceType.KING, Color.WHITE)
        self.set_piece(0, 2, 0, PieceType.PAWN, Color.WHITE)
        self.set_piece(0, 3, 0, PieceType.PAWN, Color.BLACK)
        self.set_piece(0, 5, 0, PieceType.ROOK, Color.BLACK)
        
        # Generate moves for BLACK
        self.state.color = Color.BLACK
        generate_legal_moves(self.state)
        
        # Check pins for WHITE
        pinned = get_pinned_pieces(self.state, Color.WHITE)
        
        self.assertEqual(len(pinned), 0, "No pieces should be pinned (enemy block)")
        print("Enemy Block Test Passed!")

    def test_pin_filtering(self):
        """Test that pinned pieces have restricted legal moves."""
        print("\nTesting Pin Filtering...")
        # Setup:
        # White King at (0,0,0)
        # White Rook at (0,2,0) (Pinned)
        # Black Rook at (0,5,0) (Pinner)
        
        self.set_piece(0, 0, 0, PieceType.KING, Color.WHITE)
        self.set_piece(0, 2, 0, PieceType.ROOK, Color.WHITE)
        self.set_piece(0, 5, 0, PieceType.ROOK, Color.BLACK)
        
        # 1. Generate moves for BLACK to populate raw moves cache (needed for pin detection)
        self.state.color = Color.BLACK
        generate_legal_moves(self.state)
        
        # 2. Generate moves for WHITE
        self.state.color = Color.WHITE
        legal_moves = generate_legal_moves(self.state)
        
        # Filter moves for the White Rook at (0,2,0)
        rook_moves = []
        for move in legal_moves:
            if np.all(move[:3] == [0, 2, 0]):
                rook_moves.append(move)
        
        rook_moves = np.array(rook_moves)
        
        # Allowed moves for Rook at (0,2,0):
        # - Move towards King: (0,1,0)
        # - Move towards Attacker: (0,3,0), (0,4,0)
        # - Capture Attacker: (0,5,0)
        # - Move away from Attacker (away from King)? No, that leaves line.
        
        # Total allowed moves: 4
        
        # Check that NO moves move off the file (e.g. (1,2,0), (0,2,1))
        for move in rook_moves:
            dest = move[3:]
            if dest[0] != 0 or dest[2] != 0:
                self.fail(f"Pinned Rook moved off the pin line to {dest}")
                
        # Check that it CAN move along the line
        destinations = rook_moves[:, 3:]
        
        can_move_towards_king = np.any(np.all(destinations == [0, 1, 0], axis=1))
        can_move_towards_attacker = np.any(np.all(destinations == [0, 3, 0], axis=1))
        can_capture = np.any(np.all(destinations == [0, 5, 0], axis=1))
        
        self.assertTrue(can_move_towards_king, "Pinned Rook should be able to move towards King")
        self.assertTrue(can_move_towards_attacker, "Pinned Rook should be able to move towards Attacker")
        self.assertTrue(can_capture, "Pinned Rook should be able to capture Attacker")
        
        print("Pin Filtering Test Passed!")

if __name__ == '__main__':
    unittest.main()
