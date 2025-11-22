
import unittest
import numpy as np
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, FLOAT_DTYPE
from training.opponents import (
    _compute_check_potential_vectorized,
    _compute_capture_rewards_vectorized,
    PriestHunterOpponent,
    PieceCaptureOpponent
)

class TestRewards(unittest.TestCase):
    def test_compute_check_potential(self):
        # Test Knight check
        to_coords = np.array([[1, 2, 0]], dtype=COORD_DTYPE)
        piece_types = np.array([PieceType.KNIGHT.value], dtype=np.int8)
        enemy_king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        
        rewards = _compute_check_potential_vectorized(
            to_coords, piece_types, enemy_king_pos, 1.5
        )
        self.assertEqual(rewards[0], 1.5)

        # Test non-check
        to_coords_safe = np.array([[3, 3, 3]], dtype=COORD_DTYPE)
        rewards_safe = _compute_check_potential_vectorized(
            to_coords_safe, piece_types, enemy_king_pos, 1.5
        )
        self.assertEqual(rewards_safe[0], 0.0)

    def test_priest_capture_reward(self):
        # Setup mock data for PriestHunter
        to_coords = np.array([[0, 0, 0]], dtype=COORD_DTYPE)
        captured_colors = np.array([Color.BLACK.value], dtype=np.int8)
        captured_types = np.array([PieceType.PRIEST.value], dtype=np.int8)
        
        # Manually invoke the logic snippet from PriestHunter (since we can't easily mock full state here without complex setup)
        # But we can test the vectorized function it uses if we modified it? 
        # Actually PriestHunter modifies the loop directly.
        
        # Let's test _compute_capture_rewards_vectorized for standard opponents
        rewards = _compute_capture_rewards_vectorized(
            to_coords, captured_colors, captured_types, Color.WHITE.value,
            priest_bonus=1.0, queen_rook_bonus=0.3
        )
        self.assertEqual(rewards[0], 1.5) # 0.5 base + 1.0 bonus

    def test_check_potential_pawn(self):
        # Pawn check (diagonal)
        to_coords = np.array([[1, 1, 1]], dtype=COORD_DTYPE)
        piece_types = np.array([PieceType.PAWN.value], dtype=np.int8)
        enemy_king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        
        rewards = _compute_check_potential_vectorized(
            to_coords, piece_types, enemy_king_pos, 1.5
        )
        self.assertEqual(rewards[0], 1.5)

if __name__ == '__main__':
    unittest.main()
