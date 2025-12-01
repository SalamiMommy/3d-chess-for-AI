
import numpy as np
import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, Color
from game3d.movement.slider_engine import _generate_all_slider_moves_batch
from game3d.movement.jump_engine import _generate_and_filter_jump_moves_batch

class TestParallelMoveGeneration(unittest.TestCase):
    def setUp(self):
        self.flattened = np.zeros(SIZE**3, dtype=np.int8)
        self.occ = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8)
        
        # Add some obstacles
        self.obstacles = [
            (2, 2, 2), (3, 3, 3), (4, 4, 4)
        ]
        for x, y, z in self.obstacles:
            idx = x + SIZE * y + SIZE_SQUARED * z
            self.flattened[idx] = Color.BLACK
            self.occ[x, y, z] = Color.BLACK

    def test_slider_parallel_vs_serial_logic(self):
        # Setup
        positions = np.array([[1, 1, 1], [5, 5, 5]], dtype=COORD_DTYPE)
        directions = np.array([[1, 1, 1], [-1, -1, -1]], dtype=COORD_DTYPE)
        max_dists = np.array([7, 7], dtype=np.int32)
        
        # Run parallel kernel
        moves, _ = _generate_all_slider_moves_batch(
            Color.WHITE, positions, directions, max_dists, self.flattened, False
        )
        
        # Verify manually
        expected_moves = []
        for i in range(len(positions)):
            px, py, pz = positions[i]
            for d in range(len(directions)):
                dx, dy, dz = directions[d]
                cx, cy, cz = px + dx, py + dy, pz + dz
                
                for _ in range(max_dists[i]):
                    if not (0 <= cx < SIZE and 0 <= cy < SIZE and 0 <= cz < SIZE):
                        break
                    
                    occ_val = self.occ[cx, cy, cz]
                    if occ_val == 0:
                        expected_moves.append([px, py, pz, cx, cy, cz])
                        cx += dx
                        cy += dy
                        cz += dz
                    elif occ_val != Color.WHITE:
                        expected_moves.append([px, py, pz, cx, cy, cz])
                        break
                    else:
                        break
                        
        expected_arr = np.array(expected_moves, dtype=COORD_DTYPE)
        
        # Sort both for comparison
        if len(moves) > 0:
            moves = moves[np.lexsort((moves[:, 5], moves[:, 4], moves[:, 3], moves[:, 2], moves[:, 1], moves[:, 0]))]
        if len(expected_arr) > 0:
            expected_arr = expected_arr[np.lexsort((expected_arr[:, 5], expected_arr[:, 4], expected_arr[:, 3], expected_arr[:, 2], expected_arr[:, 1], expected_arr[:, 0]))]
            
        np.testing.assert_array_equal(moves, expected_arr)

    def test_jump_parallel_vs_serial_logic(self):
        # Setup
        positions = np.array([[1, 1, 1], [3, 3, 3]], dtype=COORD_DTYPE)
        directions = np.array([[1, 2, 0], [2, 1, 0]], dtype=COORD_DTYPE)
        
        # Run parallel kernel
        moves = _generate_and_filter_jump_moves_batch(
            positions, directions, self.occ, True, Color.WHITE
        )
        
        # Verify manually
        expected_moves = []
        for i in range(len(positions)):
            px, py, pz = positions[i]
            for d in range(len(directions)):
                dx, dy, dz = directions[d]
                tx, ty, tz = px + dx, py + dy, pz + dz
                
                if 0 <= tx < SIZE and 0 <= ty < SIZE and 0 <= tz < SIZE:
                    occ_val = self.occ[tx, ty, tz]
                    if occ_val == 0 or occ_val != Color.WHITE:
                        expected_moves.append([px, py, pz, tx, ty, tz])
                        
        expected_arr = np.array(expected_moves, dtype=COORD_DTYPE)
        
        # Sort both for comparison
        if len(moves) > 0:
            moves = moves[np.lexsort((moves[:, 5], moves[:, 4], moves[:, 3], moves[:, 2], moves[:, 1], moves[:, 0]))]
        if len(expected_arr) > 0:
            expected_arr = expected_arr[np.lexsort((expected_arr[:, 5], expected_arr[:, 4], expected_arr[:, 3], expected_arr[:, 2], expected_arr[:, 1], expected_arr[:, 0]))]
            
        np.testing.assert_array_equal(moves, expected_arr)

if __name__ == '__main__':
    unittest.main()
