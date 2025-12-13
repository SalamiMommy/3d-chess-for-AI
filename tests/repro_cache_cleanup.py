
import unittest
import numpy as np
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.common.coord_utils import coords_to_keys
from game3d.game.turnmove import make_move

class TestCacheCleanup(unittest.TestCase):
    def test_cache_cleanup_on_move(self):
        # 1. Setup Board
        gs = GameState(Board.empty(), Color.WHITE)
        
        # CLEAR BOARD explicitly
        # We can use cache manager rebuild with empty arrays
        empty_coords = np.empty((0, 3), dtype=COORD_DTYPE)
        empty_types = np.array([], dtype=np.int8)
        empty_colors = np.array([], dtype=np.int8)
        gs.cache_manager.occupancy_cache.rebuild(empty_coords, empty_types, empty_colors)
        
        # Place Rook at (4,4,4) and King at (0,0,0)
        start_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
        king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        
        # Prepare batch update
        coords = np.vstack([start_pos, king_pos])
        pieces = np.array([
            [PieceType.ROOK, Color.WHITE],
            [PieceType.KING, Color.WHITE]
        ], dtype=np.int8)
        
        gs.cache_manager.occupancy_cache.batch_set_positions(coords, pieces)
        
        # 2. Generate Moves
        import game3d.movement.generator as gen_module
        gen_module.initialize_generator()
        
        moves = gen_module.generate_legal_moves(gs)
        
        # Verify cache
        move_cache = gs.cache_manager.move_cache
        start_key = int(coords_to_keys(start_pos.reshape(1,3))[0])
        # Color=White(1) -> Index 0
        expected_key = start_key 
        # (Technically | (0 << 30) which is 0)
        
        print(f"Checking for key: {expected_key} (start_key={start_key})")
        print(f"Current keys in cache: {list(move_cache._piece_moves_cache.keys())}")
        
        is_in_cache = expected_key in move_cache._piece_moves_cache
        self.assertTrue(is_in_cache, "Piece moves should be in cache after generation")
        
        # 3. Make Move (4,4,4) -> (4,4,5)
        move_arr = np.array([4, 4, 4, 4, 4, 5], dtype=np.int8) 
        
        gs = make_move(gs, move_arr)
        
        # 4. Check Cache Cleanup
        # The entry for (4,4,4) MUST be gone.
        is_still_in_cache = expected_key in move_cache._piece_moves_cache
        
        if is_still_in_cache:
            print(f"FAILURE: Old cache entry persisted! Keys: {list(move_cache._piece_moves_cache.keys())}")
        else:
            print("SUCCESS: Old cache entry removed.")
            
        self.assertFalse(is_still_in_cache, "Old piece cache entry should be removed after move")

if __name__ == "__main__":
    unittest.main()
