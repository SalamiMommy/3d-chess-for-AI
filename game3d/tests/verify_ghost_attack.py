
import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, SIZE, COLOR_WHITE, COLOR_BLACK
from game3d.attacks.check import king_in_check, get_attackers
from game3d.game.terminal import is_game_over

class TestGhostAttackRepro(unittest.TestCase):
    def setUp(self):
        self.state = GameState.from_startpos()
        self.board = self.state.board
        self.cache_manager = self.state.cache_manager
        
        # Clear board
        self.cache_manager.occupancy_cache.rebuild(
            np.empty((0, 3), dtype=np.int16),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8)
        )
        self.cache_manager.move_cache.invalidate()

    def test_ghost_attack_after_capture(self):
        """
        Reproduction of Ghost Attack:
        1. Place White King.
        2. Place Black Rook checking the King.
        3. Verify Check.
        4. Place White Knight and capture the Rook.
        5. Verify NO Check.
        """
        print("\n--- Test Ghost Attack After Capture ---")
        
        # 1. Place White King at (4,4,4)
        wk_pos = np.array([4, 4, 4], dtype=np.int16)
        self.cache_manager.occupancy_cache.set_position(wk_pos, np.array([PieceType.KING, Color.WHITE]))
        
        # 2. Place Black Rook at (4,4,0) - Attacking King
        br_pos = np.array([4, 4, 0], dtype=np.int16)
        self.cache_manager.occupancy_cache.set_position(br_pos, np.array([PieceType.ROOK, Color.BLACK]))
        
        # Refresh cache to generate attacks
        self.state._legal_moves_cache = None
        
        # CRITICAL: Force Black move generation to populate _attack_matrix
        # White's legal_moves only populates opponent pseudolegal cache via fallback, 
        # which skips matrix update. We need full pipeline.
        offset_original_color = self.state.color
        self.state.color = Color.BLACK
        from game3d.movement.generator import generate_legal_moves
        generate_legal_moves(self.state)
        self.state.color = offset_original_color
        
        # Trigger move generation for White
        self.state.legal_moves
        
        # 3. Verify Check
        is_check = king_in_check(self.board, Color.WHITE, Color.WHITE, self.cache_manager)
        attackers = get_attackers(self.state)
        print(f"Initial Check: {is_check}")
        print(f"Attackers: {attackers}")
        
        self.assertTrue(is_check, "King should be in check")
        self.assertTrue(len(attackers) > 0, "Should have attackers")
        self.assertIn("ROOK", str(attackers), "Rook should be attacking")
        
        # 4. Capture the Rook with a White Knight
        # Place Knight at (3,2,0) then move it to (4,4,0)
        wn_pos = np.array([3, 2, 0], dtype=np.int16)
        self.cache_manager.occupancy_cache.set_position(wn_pos, np.array([PieceType.KNIGHT, Color.WHITE]))
        
        print("Capturing Rook with Knight (Move Knight to 4,4,0)...")
        self.cache_manager.occupancy_cache.set_position(wn_pos, None) # Clear Knight old pos
        self.cache_manager.occupancy_cache.set_position(br_pos, np.array([PieceType.KNIGHT, Color.WHITE])) # Knight captures Rook
        
        # Update GameState internals
        self.state._legal_moves_cache = None
        # Simulate what apply_move does: invalidate legal moves cache
        self.cache_manager.move_cache.invalidate_legal_moves()
        # Also invalidate pseudolegal to force get_incremental_state (which triggers cleanup)
        self.cache_manager.move_cache.invalidate_pseudolegal_moves(Color.BLACK)
        self.state.color = Color.BLACK # Turn passes
        
        # NOW: We check if WHITE King is still in check (it shouldn't be, Rook is gone)
        # But Ghost Rook at (4,4,0) attacks King at (4,4,4).
        
        print("Generating moves for Black (should trigger incremental update)...")
        self.state.color = Color.BLACK
        from game3d.movement.generator import generate_legal_moves
        generate_legal_moves(self.state)
        
        # Now switch back to White and check check status
        print("Checking White King Safety...")
        # King is still at wk_pos (4,4,4)
        
        # We check is_under_attack for White King position
        is_under_attack_bit = self.cache_manager.move_cache.is_under_attack(wk_pos, Color.WHITE)
        print(f"Bitboard says is_under_attack: {is_under_attack_bit}")
        
        # And get attackers list
        dummy_state = GameState.from_startpos()
        dummy_state.board = self.board
        dummy_state.cache_manager = self.cache_manager
        dummy_state.color = Color.WHITE
        attackers = get_attackers(dummy_state)
        print(f"Attackers list: {attackers}")
        
        # Assertion
        self.assertFalse(is_under_attack_bit, "Ghost Attack! Bitboard should be False after Rook capture")
        self.assertEqual(len(attackers), 0, "Should have 0 attackers")

if __name__ == '__main__':
    unittest.main()
