
import unittest
import unittest
import numpy as np
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType
from game3d.attacks.check import move_would_leave_king_in_check
from game3d.movement.generator import initialize_generator, _generator

class TestMoveSimulation(unittest.TestCase):
    def setUp(self):
        self.state = GameState.from_startpos()
        self.cache_manager = self.state.cache_manager
        self.occ_cache = self.cache_manager.occupancy_cache
        self.move_cache = self.cache_manager.move_cache
        
        # Initialize generator
        from game3d.movement.generator import initialize_generator, _generator
        if _generator is None:
            initialize_generator()
        self.move_generator = _generator
            
    def _clear_board(self):
        """Helper to clear all pieces from board."""
        for color in [Color.WHITE, Color.BLACK]:
            coords = self.occ_cache.get_positions(color)
            for i in range(coords.shape[0]):
                self.occ_cache.set_position(coords[i], np.array([0, 0])) # Remove
        
        # CRITICAL: Rebuild AuraCache because set_position does not update it
        self.cache_manager.aura_cache.rebuild_from_board(self.occ_cache)

    def _refresh_black_moves(self):
        """Helper to ensure Black moves are cached so we hit optimized path."""
        # Force cache invalidation by bumping generation
        if hasattr(self.state.cache_manager.board, 'generation'):
            self.state.cache_manager.board.generation += 1
            
        prev = self.state.color
        self.state.color = Color.BLACK
        from game3d.movement.generator import _generator
        _generator.refresh_pseudolegal_moves(self.state)
        self.state.color = prev

    def test_pinned_piece(self):
        """Test that a pinned piece cannot move."""
        self._clear_board()
        
        # 2. Setup Pin: King(0,0,0) - Pawn(0,1,0) - Rook(0,5,0, Black)
        self.occ_cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
        self.occ_cache.set_position(np.array([0,1,0]), np.array([PieceType.PAWN, Color.WHITE]))
        self.occ_cache.set_position(np.array([0,5,0]), np.array([PieceType.ROOK, Color.BLACK]))
        
        # Update King Cache
        self.occ_cache._king_positions[0] = np.array([0,0,0])
        self.state.color = Color.WHITE
        
        # Force refresh black moves (attacker)
        self._refresh_black_moves()
        
        # 3. Try to move Pawn (0,1,0) -> (1,1,0) (Illegal, exposing check)
        move = np.array([0,1,0, 1,1,0])
        is_unsafe = move_would_leave_king_in_check(self.state, move)
        self.assertTrue(is_unsafe, "Pinned pawn moving aside should be unsafe")
        
        # 4. Try to move Pawn (0,1,0) -> (0,2,0) (Legal, moving along pin line)
        move_legal = np.array([0,1,0, 0,2,0])
        is_unsafe = move_would_leave_king_in_check(self.state, move_legal)
        self.assertFalse(is_unsafe, "Pinned pawn moving along attack line should be safe (blocking)")
        
        # 5. Capture Attacker: Pawn (0,1,0) -> (0,5,0) (Legal if reachable)
        move_capture = np.array([0,1,0, 0,5,0])
        is_unsafe = move_would_leave_king_in_check(self.state, move_capture)
        self.assertFalse(is_unsafe, "Capturing the pinner should be safe")

    def test_king_move(self):
        """Test King moving into/out of check."""
        self._clear_board()
        
        # Setup: King(0,0,0), Rook(0,2,0, Black) -> Check along Y
        self.occ_cache._king_positions[0] = np.array([0,0,0])
        self.state.color = Color.WHITE
        
        self.occ_cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
        self.occ_cache.set_position(np.array([0,2,0]), np.array([PieceType.ROOK, Color.BLACK]))
        
        self._refresh_black_moves()
        
        # 1. Move King along Y line (0,0,0 -> 0,1,0) - Still checked by Rook at 0,2,0
        # Note: 0,1,0 is attacked by Rook at 0,2,0.
        # Debug Occupancy
        black_pieces = self.occ_cache.get_positions(Color.BLACK)
        print(f"DEBUG: Black Pieces: {black_pieces.tolist()}. Count: {len(black_pieces)}")
        print(f"DEBUG: Positions Dirty: {self.occ_cache._positions_dirty}")
        
        # Debug Attack Matrix
        cache = self.state.cache_manager.move_cache
        flat_idx = 9 # 0,1,0
        matrix = cache._attack_matrix[Color.BLACK - 1][flat_idx]
        is_attacked = np.any(matrix > 0)
        print(f"DEBUG TEST: Square 0,1,0 attacked directly? {is_attacked}. Matrix: {matrix}")
        
        # Inspect cached moves
        moves = cache._pseudolegal_moves_cache[Color.BLACK - 1]
        print(f"DEBUG: Black moves count: {len(moves)}")
        found = False
        for i in range(len(moves)):
            m = moves[i]
            # m is record (from_x, from_y, from_z, to_x, to_y, to_z)
            if m[3]==0 and m[4]==1 and m[5]==0:
                print(f"DEBUG: Found move to 0,1,0! From {m[0]},{m[1]},{m[2]}")
                found = True
        if not found:
            print("DEBUG: NO MOVE to 0,1,0 found!")
        
        move_unsafe = np.array([0,0,0, 0,1,0])
        is_unsafe = move_would_leave_king_in_check(self.state, move_unsafe)
        self.assertTrue(is_unsafe, "King moving towards rook along file should still be in check (or captured)")
        
        # 2. Move King aside (0,0,0 -> 1,0,0) - Safe
        move_safe = np.array([0,0,0, 1,0,0])
        is_unsafe = move_would_leave_king_in_check(self.state, move_safe)
        self.assertFalse(is_unsafe, "King moving out of line of fire should be safe")
        
        # 3. Move King to capture Rook (0,0,0 -> 0,2,0) - Safe
        move_capture = np.array([0,0,0, 0,2,0])
        is_unsafe = move_would_leave_king_in_check(self.state, move_capture)
        self.assertFalse(is_unsafe, "King capturing unprotected rook should be safe")

    def test_swapper_move(self):
        """Test Swapper move."""
        self._clear_board()
        
        # Setup: King(0,0,0), Swapper(0,1,0), Pawn(1,1,0), Rook(0,5,0, Black)
        self.occ_cache._king_positions[0] = np.array([0,0,0])
        self.state.color = Color.WHITE
        
        self.occ_cache.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
        self.occ_cache.set_position(np.array([0,1,0]), np.array([PieceType.SWAPPER, Color.WHITE]))
        self.occ_cache.set_position(np.array([1,1,0]), np.array([PieceType.PAWN, Color.WHITE]))
        self.occ_cache.set_position(np.array([0,5,0]), np.array([PieceType.ROOK, Color.BLACK]))
        
        self._refresh_black_moves()
        
        # 1. Swapper swaps with Pawn (0,1,0 -> 1,1,0). 
        move_swap = np.array([0,1,0, 1,1,0])
        is_unsafe = move_would_leave_king_in_check(self.state, move_swap)
        self.assertFalse(is_unsafe, "Swapper swapping with friendly piece blocking check should be safe")
        
        # 2. Swapper moves aside: 0,1,0 -> 2,2,2
        move_aside = np.array([0,1,0, 2,2,2])
        is_unsafe = move_would_leave_king_in_check(self.state, move_aside)
        self.assertTrue(is_unsafe, "Swapper moving out of pin should be unsafe")

if __name__ == '__main__':
    unittest.main()
