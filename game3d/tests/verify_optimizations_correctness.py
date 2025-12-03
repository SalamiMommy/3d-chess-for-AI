
import numpy as np
import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import COORD_DTYPE, PieceType, Color, SIZE
from game3d.pieces.pieces.archer import generate_archer_moves
from game3d.pieces.pieces.friendlytp import generate_friendlytp_moves
from game3d.game.gamestate import GameState
from game3d.cache.manager import OptimizedCacheManager

class TestOptimizations(unittest.TestCase):
    def setUp(self):
        self.state = GameState.from_startpos()
        self.cache_manager = self.state.cache_manager
        self.occ = self.cache_manager.occupancy_cache
        
        # Clear board
        empty_coords = np.zeros((0, 3), dtype=COORD_DTYPE)
        empty_types = np.zeros(0, dtype=np.int8)
        empty_colors = np.zeros(0, dtype=np.int8)
        self.occ.rebuild(empty_coords, empty_types, empty_colors)

    def test_archer_moves(self):
        # Setup: Archer at [4, 4, 4]
        # Enemy at [4, 4, 6] (dist 2, valid shot)
        # Friendly at [6, 4, 4] (dist 2, invalid shot)
        # Empty at [4, 6, 4] (dist 2, invalid shot - must be enemy)
        
        archer_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
        enemy_pos = np.array([4, 4, 6], dtype=COORD_DTYPE)
        friendly_pos = np.array([6, 4, 4], dtype=COORD_DTYPE)
        
        coords = np.array([archer_pos, enemy_pos, friendly_pos], dtype=COORD_DTYPE)
        types = np.array([PieceType.ARCHER, PieceType.PAWN, PieceType.PAWN], dtype=np.int8)
        colors = np.array([Color.WHITE, Color.BLACK, Color.WHITE], dtype=np.int8)
        
        self.occ.rebuild(coords, types, colors)
        
        moves = generate_archer_moves(self.cache_manager, Color.WHITE, archer_pos)
        
        # Check shots
        # Move format: [fx, fy, fz, tx, ty, tz]
        shots = []
        for m in moves:
            # Shot if distance > 1 (approx check, king moves are dist 1)
            dist = np.max(np.abs(m[3:] - m[:3]))
            if dist > 1:
                shots.append(m)
                
        self.assertEqual(len(shots), 1, "Should have exactly 1 shot")
        self.assertTrue(np.array_equal(shots[0][3:], enemy_pos), "Shot should target enemy")
        
        # Check king moves (should be 26 - 2 occupied = 24? No, friendly blocks, enemy captures)
        # King moves are 1-step.
        # [4,4,6] is dist 2.
        # [6,4,4] is dist 2.
        # So immediate neighbors are empty.
        # Should have 26 king moves.
        
        king_moves = [m for m in moves if np.max(np.abs(m[3:] - m[:3])) == 1]
        self.assertEqual(len(king_moves), 26, "Should have 26 king moves into empty space")

    def test_friendlytp_moves(self):
        # Setup: FTP at [0, 0, 0]
        # Friendly piece at [8, 8, 8]
        # Neighbors of [8, 8, 8]: [7, 8, 8], [8, 7, 8], etc.
        
        ftp_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
        friendly_pos = np.array([8, 8, 8], dtype=COORD_DTYPE)
        
        coords = np.array([ftp_pos, friendly_pos], dtype=COORD_DTYPE)
        types = np.array([PieceType.FRIENDLYTELEPORTER, PieceType.PAWN], dtype=np.int8)
        colors = np.array([Color.WHITE, Color.WHITE], dtype=np.int8)
        
        self.occ.rebuild(coords, types, colors)
        
        moves = generate_friendlytp_moves(self.cache_manager, Color.WHITE, ftp_pos)
        
        # Check teleports
        # Teleport if distance > 1
        teleports = []
        for m in moves:
            dist = np.max(np.abs(m[3:] - m[:3]))
            if dist > 1:
                teleports.append(m)
        
        # Neighbors of [8,8,8] (corner)
        # Valid: [7,8,8], [8,7,8], [8,8,7], [7,7,8], [7,8,7], [8,7,7], [7,7,7]
        # Total 7 neighbors for a corner.
        
        self.assertGreater(len(teleports), 0, "Should have teleports")
        
        # Verify all teleports land near friendly_pos
        for t in teleports:
            dest = t[3:]
            dist_to_friend = np.max(np.abs(dest - friendly_pos))
            self.assertEqual(dist_to_friend, 1, "Teleport must land adjacent to friendly piece")

if __name__ == '__main__':
    unittest.main()
