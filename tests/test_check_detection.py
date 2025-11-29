
import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

# Add current directory to path so we can import game3d
sys.path.append(os.getcwd())

from game3d.common.shared_types import (
    Color, PieceType, SIZE, MOVE_DTYPE, COORD_DTYPE
)
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.cache.caches.movecache import MoveCache
from game3d.attacks.check import king_in_check

class MockCacheManager:
    def __init__(self):
        self.occupancy_cache = None
        self.move_cache = None
        self.board = None

class MockBoard:
    def __init__(self):
        self.cache_manager = MockCacheManager()
        self.generation = 0

class TestCheckDetection(unittest.TestCase):
    def setUp(self):
        self.board = MockBoard()
        self.occ_cache = OccupancyCache()
        self.move_cache = MoveCache(self.board.cache_manager)
        
        # Wire up the mock cache manager
        self.board.cache_manager.occupancy_cache = self.occ_cache
        self.board.cache_manager.move_cache = self.move_cache
        self.board.cache_manager.board = self.board

    def test_king_under_attack_no_priests(self):
        """Test that check is detected when King is under attack and has no priests."""
        print("\nRunning test_king_under_attack_no_priests...")
        # 1. Place White King at (0, 0, 0)
        king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        self.occ_cache.set_position(king_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
        
        # 2. Place Black Rook at (0, 0, 5)
        attacker_pos = np.array([0, 0, 5], dtype=COORD_DTYPE)
        self.occ_cache.set_position(attacker_pos, np.array([PieceType.ROOK.value, Color.BLACK.value]))
        
        # 3. Mock MoveCache to say Black attacks (0, 0, 0)
        # The check logic calls _get_attacked_squares_from_move_cache -> move_cache.get_raw_moves
        # We need to mock get_raw_moves to return a move ending at (0, 0, 0)
        
        # Move format: [from_x, from_y, from_z, to_x, to_y, to_z]
        mock_attack_move = np.zeros(1, dtype=MOVE_DTYPE)
        mock_attack_move[0]['from_x'] = 0
        mock_attack_move[0]['from_y'] = 0
        mock_attack_move[0]['from_z'] = 5
        mock_attack_move[0]['to_x'] = 0
        mock_attack_move[0]['to_y'] = 0
        mock_attack_move[0]['to_z'] = 0
        mock_attack_move[0]['to_z'] = 0
        # Fix: check.py uses get_pseudolegal_moves, so we must mock that cache level
        self.move_cache.store_pseudolegal_moves(Color.BLACK, mock_attack_move)
        
        # 4. Verify Check
        is_check = king_in_check(self.board, Color.WHITE, Color.WHITE, cache=self.board.cache_manager)
        print(f"  Result: is_check={is_check}")
        self.assertTrue(is_check, "King should be in check when under attack with 0 priests")

    def test_king_under_attack_with_priests(self):
        """Test that check is NOT detected when King is under attack but has priests."""
        print("\nRunning test_king_under_attack_with_priests...")
        # 1. Place White King at (0, 0, 0)
        king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        self.occ_cache.set_position(king_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
        
        # 2. Place Black Rook at (0, 0, 5)
        attacker_pos = np.array([0, 0, 5], dtype=COORD_DTYPE)
        self.occ_cache.set_position(attacker_pos, np.array([PieceType.ROOK.value, Color.BLACK.value]))
        
        # 3. Place White Priest at (1, 1, 1)
        priest_pos = np.array([1, 1, 1], dtype=COORD_DTYPE)
        self.occ_cache.set_position(priest_pos, np.array([PieceType.PRIEST.value, Color.WHITE.value]))
        
        # 4. Mock MoveCache to say Black attacks (0, 0, 0)
        mock_attack_move = np.zeros(1, dtype=MOVE_DTYPE)
        mock_attack_move[0]['from_x'] = 0
        mock_attack_move[0]['from_y'] = 0
        mock_attack_move[0]['from_z'] = 5
        mock_attack_move[0]['to_x'] = 0
        mock_attack_move[0]['to_y'] = 0
        mock_attack_move[0]['to_z'] = 0
        mock_attack_move[0]['to_z'] = 0
        # Fix: check.py uses get_pseudolegal_moves, so we must mock that cache level
        self.move_cache.store_pseudolegal_moves(Color.BLACK, mock_attack_move)
        
        # 5. Verify NO Check
        is_check = king_in_check(self.board, Color.WHITE, Color.WHITE, cache=self.board.cache_manager)
        print(f"  Result: is_check={is_check}")
        self.assertFalse(is_check, "King should NOT be in check when priests are alive")

    def test_king_safe_no_priests(self):
        """Test that no check is detected when King is safe and has no priests."""
        print("\nRunning test_king_safe_no_priests...")
        # 1. Place White King at (0, 0, 0)
        king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        self.occ_cache.set_position(king_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
        
        # 2. No attackers
        # 2. No attackers
        self.move_cache.store_pseudolegal_moves(Color.BLACK, np.empty((0, 6), dtype=MOVE_DTYPE))
        
        # 3. Verify NO Check
        is_check = king_in_check(self.board, Color.WHITE, Color.WHITE, cache=self.board.cache_manager)
        print(f"  Result: is_check={is_check}")
        self.assertFalse(is_check, "King should not be in check when no pieces are attacking")

if __name__ == '__main__':
    unittest.main()
