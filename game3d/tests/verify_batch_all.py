
import numpy as np
import unittest
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import SIZE, Color, PieceType, COORD_DTYPE
from game3d.pieces.pieces.rook import generate_rook_moves
from game3d.pieces.pieces.bishop import generate_bishop_moves
from game3d.pieces.pieces.queen import generate_queen_moves
from game3d.pieces.pieces.bomb import generate_bomb_moves
from game3d.pieces.pieces.archer import generate_archer_moves
from game3d.pieces.pieces.swapper import generate_swapper_moves
from game3d.pieces.pieces.friendlytp import generate_friendlytp_moves

class MockBoard:
    def __init__(self):
        self.generation = 0
    def get_initial_setup(self):
        return (np.empty((0, 3)), np.empty(0), np.empty(0))

class TestBatchAll(unittest.TestCase):
    def setUp(self):
        self.cache_manager = OptimizedCacheManager(MockBoard())
        self.cache_manager.occupancy_cache.clear()

    def _test_batch_vs_sequential(self, generator, piece_type):
        # Place 2 pieces
        pos1 = np.array([4, 4, 4], dtype=COORD_DTYPE)
        pos2 = np.array([0, 0, 0], dtype=COORD_DTYPE)
        
        # Set positions in cache so they don't capture each other if friendly
        self.cache_manager.occupancy_cache.set_position(pos1, np.array([piece_type.value, Color.WHITE.value]))
        self.cache_manager.occupancy_cache.set_position(pos2, np.array([piece_type.value, Color.WHITE.value]))
        
        # Sequential
        moves1 = generator(self.cache_manager, Color.WHITE, pos1)
        moves2 = generator(self.cache_manager, Color.WHITE, pos2)
        
        # Batch
        batch_pos = np.vstack([pos1, pos2])
        batch_moves = generator(self.cache_manager, Color.WHITE, batch_pos)
        
        # Verify counts
        self.assertEqual(len(batch_moves), len(moves1) + len(moves2), f"Count mismatch for {piece_type}")
        
        # Verify content
        expected = np.vstack([moves1, moves2]) if len(moves1) > 0 or len(moves2) > 0 else np.empty((0, 6), dtype=COORD_DTYPE)
        
        if len(expected) > 0:
            expected_sorted = expected[np.lexsort(expected.T[::-1])]
            batch_sorted = batch_moves[np.lexsort(batch_moves.T[::-1])]
            np.testing.assert_array_equal(batch_sorted, expected_sorted, err_msg=f"Content mismatch for {piece_type}")
        else:
            self.assertEqual(len(batch_moves), 0)
            
        print(f"✅ {piece_type.name} batching verified")

    def test_rook(self):
        self._test_batch_vs_sequential(generate_rook_moves, PieceType.ROOK)

    def test_bishop(self):
        self._test_batch_vs_sequential(generate_bishop_moves, PieceType.BISHOP)

    def test_queen(self):
        self._test_batch_vs_sequential(generate_queen_moves, PieceType.QUEEN)

    def test_bomb(self):
        # Bomb needs enemies for detonation
        enemy_pos = np.array([5, 5, 5], dtype=COORD_DTYPE) # Near 4,4,4
        self.cache_manager.occupancy_cache.set_position(enemy_pos, np.array([PieceType.PAWN.value, Color.BLACK.value]))
        self._test_batch_vs_sequential(generate_bomb_moves, PieceType.BOMB)

    def test_archer(self):
        # Archer needs enemies for shots
        enemy_pos = np.array([6, 4, 4], dtype=COORD_DTYPE) # Distance 2 from 4,4,4
        self.cache_manager.occupancy_cache.set_position(enemy_pos, np.array([PieceType.PAWN.value, Color.BLACK.value]))
        self._test_batch_vs_sequential(generate_archer_moves, PieceType.ARCHER)

    def test_swapper(self):
        # Swapper needs friendly pieces
        friendly_pos = np.array([2, 2, 2], dtype=COORD_DTYPE)
        self.cache_manager.occupancy_cache.set_position(friendly_pos, np.array([PieceType.PAWN.value, Color.WHITE.value]))
        self._test_batch_vs_sequential(generate_swapper_moves, PieceType.SWAPPER)

    def test_friendlytp(self):
        # FriendlyTP needs friendly pieces
        friendly_pos = np.array([2, 2, 2], dtype=COORD_DTYPE)
        self.cache_manager.occupancy_cache.set_position(friendly_pos, np.array([PieceType.PAWN.value, Color.WHITE.value]))
        self._test_batch_vs_sequential(generate_friendlytp_moves, PieceType.FRIENDLYTELEPORTER)

if __name__ == "__main__":
    unittest.main()
