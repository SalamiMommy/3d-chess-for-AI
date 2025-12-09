
import unittest
import numpy as np
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, PIECE_TYPE_DTYPE
from game3d.pieces.pieces.wall import is_wall_anchor
from game3d.game import turnmove
from game3d.movement.movepiece import Move

class TestSwapperWallCorruption(unittest.TestCase):
    def setUp(self):
        self.board = Board()
        self.cache_manager = OptimizedCacheManager(self.board)
        self.gs = GameState(self.board, Color.WHITE, self.cache_manager)
        
        # Clear board
        self.cache_manager.occupancy_cache.set_position(np.array([0,0,0], dtype=COORD_DTYPE), None)
        # Actually clear everything would be better but expensive.
        # Let's just place our pieces.

    def test_swapper_swaps_with_wall(self):
        """Test that swapping a Swapper with a Wall corrupts the walls"""
        
        # 1. Place a Wall at [0, 0, 0] (Anchor)
        # Wall occupies (0,0,0), (1,0,0), (0,1,0), (1,1,0)
        wall_parts = [
            [0,0,0], [1,0,0], [0,1,0], [1,1,0]
        ]
        wall_data = np.array([PieceType.WALL, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
        
        for p in wall_parts:
            self.gs.cache_manager.occupancy_cache.set_position(np.array(p, dtype=COORD_DTYPE), wall_data)
            
        # 2. Place a Swapper at [3, 3, 0] (Friendly)
        swapper_pos = np.array([3, 3, 0], dtype=COORD_DTYPE)
        swapper_data = np.array([PieceType.SWAPPER, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
        self.gs.cache_manager.occupancy_cache.set_position(swapper_pos, swapper_data)
        
        # 3. Create a Swap Move: Swapper [3,3,0] -> Wall Anchor [0,0,0]
        # Note: Swapper can swap with any friendly piece.
        move = Move(swapper_pos, np.array([0,0,0], dtype=COORD_DTYPE))
        
        # 4. Execute Move (Using turnmove.make_move directly or via validation)
        print("Attempting to swap Swapper with Wall Anchor...")
        
        try:
            # We use make_move_trusted to simulate the engine executing a generated move
            # If generator allows it, this will execute.
            # In unit test, we forcibly execute it to see result.
            new_gs = turnmove.make_move(self.gs, np.concatenate([move.from_coord, move.to_coord]))
            
            # 5. Check State
            # Original anchor [0,0,0] should now be Swapper
            p0 = new_gs.cache_manager.occupancy_cache.get(np.array([0,0,0], dtype=COORD_DTYPE))
            print(f"Piece at [0,0,0]: {p0}")
            self.assertEqual(p0['piece_type'], PieceType.SWAPPER)
            
            # Swapper Start [3,3,0] should now be... Wall?
            p_swap = new_gs.cache_manager.occupancy_cache.get(swapper_pos)
            print(f"Piece at {swapper_pos}: {p_swap}")
            self.assertEqual(p_swap['piece_type'], PieceType.WALL)
            
            # BUT the other wall parts should still be there!
            p1 = new_gs.cache_manager.occupancy_cache.get(np.array([1,0,0], dtype=COORD_DTYPE))
            print(f"Piece at [1,0,0]: {p1}")
            self.assertEqual(p1['piece_type'], PieceType.WALL)
            
            # This means we have a fractured wall.
            # 3 parts at original loc, 1 part at [3,3,0].
            
            # And [3,3,0] will be considered an anchor because it potentially has no left/up wall neighbors
            # If [3,3,0] is near edge (e.g. Size-1), it validates as Invalid Anchor.
            
            # Let's verify if [3,3,0] thinks it is an anchor
            is_anchor = is_wall_anchor(swapper_pos.reshape(1,3), new_gs.cache_manager)
            print(f"Is {swapper_pos} considered a wall anchor? {is_anchor}")
            self.assertTrue(is_anchor)

            print("Move succeeded (unexpectedly) and state is likely corrupted.")

        except Exception as e:
            print(f"Caught expected exception: {e}")
            # If it raises "Invalid swap", then it's already fixed/blocked.

if __name__ == '__main__':
    unittest.main()
