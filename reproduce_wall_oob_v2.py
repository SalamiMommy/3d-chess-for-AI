
import numpy as np
from game3d.common.shared_types import SIZE, PieceType, Color
from game3d.pieces.pieces.wall import generate_wall_moves
from game3d.cache.manager import OptimizedCacheManager
from game3d.board.board import Board

# Mock Cache Manager
class MockOccupancyCache:
    def __init__(self):
        self._occ = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8)
        self._occ_types = np.zeros((SIZE, SIZE, SIZE), dtype=np.int8) # 0 is EMPTY
    
    def batch_get_attributes(self, coords):
        # Mock implementation
        types = []
        for coord in coords:
            x, y, z = coord
            if 0 <= x < SIZE and 0 <= y < SIZE and 0 <= z < SIZE:
                types.append(self._occ_types[x, y, z])
            else:
                types.append(0) # EMPTY
        return None, np.array(types)

class MockAuraCache:
    def __init__(self):
        self._buffed_squares = np.zeros((SIZE, SIZE, SIZE), dtype=bool)

class MockCacheManager:
    def __init__(self):
        self.occupancy_cache = MockOccupancyCache()
        self.consolidated_aura_cache = MockAuraCache()

def test_wall_oob():
    print(f"Testing Wall OOB with SIZE={SIZE}")
    cache_manager = MockCacheManager()
    
    # Place a wall at [8, 2, 2] and buff it
    cache_manager.occupancy_cache._occ_types[8, 2, 2] = PieceType.WALL
    cache_manager.consolidated_aura_cache._buffed_squares[8, 2, 2] = True
    
    pos = np.array([[8, 2, 2]], dtype=np.int16)
    
    try:
        moves = generate_wall_moves(cache_manager, Color.WHITE, pos)
        print(f"Generated {len(moves)} moves")
        for m in moves:
            print(f"Move: {m}")
            if m[0] >= SIZE - 1 or m[3] >= SIZE - 1:
                print("❌ OOB Move detected!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_wall_oob()
