
import numpy as np
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import SIZE, Color, PieceType, COORD_DTYPE
from game3d.pieces.pieces.swapper import generate_swapper_moves

class MockBoard:
    def __init__(self):
        self.generation = 0
    def get_initial_setup(self):
        return (np.empty((0, 3)), np.empty(0), np.empty(0))

def debug_swapper():
    cache_manager = OptimizedCacheManager(MockBoard())
    cache_manager.occupancy_cache.clear()
    
    pos1 = np.array([4, 4, 4], dtype=COORD_DTYPE)
    pos2 = np.array([0, 0, 0], dtype=COORD_DTYPE)
    
    cache_manager.occupancy_cache.set_position(pos1, np.array([PieceType.SWAPPER.value, Color.WHITE.value]))
    cache_manager.occupancy_cache.set_position(pos2, np.array([PieceType.SWAPPER.value, Color.WHITE.value]))
    
    # Friendly piece
    friendly_pos = np.array([2, 2, 2], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(friendly_pos, np.array([PieceType.PAWN.value, Color.WHITE.value]))
    
    print("Sequential 1:")
    moves1 = generate_swapper_moves(cache_manager, Color.WHITE, pos1)
    print(f"Moves1 count: {len(moves1)}")
    
    print("Sequential 2:")
    moves2 = generate_swapper_moves(cache_manager, Color.WHITE, pos2)
    print(f"Moves2 count: {len(moves2)}")
    
    print("Batch:")
    batch_pos = np.vstack([pos1, pos2])
    batch_moves = generate_swapper_moves(cache_manager, Color.WHITE, batch_pos)
    print(f"Batch moves count: {len(batch_moves)}")
    
    if len(batch_moves) != len(moves1) + len(moves2):
        print("MISMATCH!")
    else:
        print("MATCH!")

if __name__ == "__main__":
    debug_swapper()
