
import numpy as np
import sys
sys.path.insert(0, '/home/salamimommy/Documents/code/3d-chess-for-AI')

from game3d.common.shared_types import COORD_DTYPE, SIZE, Color, PieceType, PAWN_START_RANK_WHITE, INDEX_DTYPE
from game3d.pieces.pieces.pawn import generate_pawn_moves
from game3d.cache.manager import OptimizedCacheManager
from game3d.cache.effectscache.trailblazecache import TrailblazeCache

def debug_pawn():
    print("=== Debugging Pawn Moves ===")
    class MockBoard:
        def __init__(self):
            self.size = SIZE
        def get_initial_setup(self):
            return (np.empty((0, 3), dtype=COORD_DTYPE), np.empty(0, dtype=int), np.empty(0, dtype=int))
    
    cm = OptimizedCacheManager(MockBoard())
    y_start = PAWN_START_RANK_WHITE
    pos = np.array([4, y_start, 4], dtype=COORD_DTYPE)
    print(f"Pawn at {pos}")
    moves = generate_pawn_moves(cm, Color.WHITE, pos)
    print(f"Moves generated: {len(moves)}")
    for m in moves:
        print(f"  {m[:3]} -> {m[3:]}")

def debug_flat_conversion():
    print("\n=== Debugging _coords_to_flat ===")
    tc = TrailblazeCache(None) # No manager needed for this test
    
    coords = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [SIZE-1, SIZE-1, SIZE-1]
    ], dtype=COORD_DTYPE)
    
    flat = tc._coords_to_flat(coords)
    print(f"Coords:\n{coords}")
    print(f"Flat indices: {flat}")
    
    # Verify manually
    expected = coords[:, 0] + SIZE * coords[:, 1] + SIZE * SIZE * coords[:, 2]
    print(f"Expected:     {expected}")
    
    if np.array_equal(flat, expected):
        print("✅ Vectorization correct")
    else:
        print("❌ Vectorization FAILED")

if __name__ == "__main__":
    debug_flat_conversion()
    debug_pawn()
