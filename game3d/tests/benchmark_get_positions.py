
import time
import numpy as np
from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import Color, PieceType, SIZE

def benchmark_get_positions():
    print(f"Benchmarking get_positions with SIZE={SIZE}...")
    
    # Initialize cache
    oc = OccupancyCache()
    
    # Setup a realistic board (32 pieces)
    # White pieces
    coords = []
    pieces = []
    
    # Pawns
    for i in range(SIZE):
        coords.append([i, 1, 0])
        pieces.append([PieceType.PAWN, Color.WHITE])
        coords.append([i, SIZE-2, SIZE-1])
        pieces.append([PieceType.PAWN, Color.BLACK])
        
    # Some other pieces
    coords.append([4, 0, 0])
    pieces.append([PieceType.KING, Color.WHITE])
    coords.append([4, SIZE-1, SIZE-1])
    pieces.append([PieceType.KING, Color.BLACK])
    
    coords = np.array(coords, dtype=np.int16)
    pieces = np.array(pieces, dtype=np.int8)
    
    oc.batch_set_positions(coords, pieces)
    
    # Benchmark loop
    # Simulate move cycle: Update 2 positions -> Get positions
    
    n_iters = 50000
    
    start_time = time.time()
    
    move_src = np.array([[4, 4, 4]], dtype=np.int16)
    move_dst = np.array([[5, 5, 5]], dtype=np.int16)
    
    # Just toggle a piece back and forth
    piece_src = np.array([[PieceType.QUEEN, Color.WHITE]], dtype=np.int8)
    piece_dst = np.array([[PieceType.QUEEN, Color.WHITE]], dtype=np.int8)
    piece_empty = np.array([[0, 0]], dtype=np.int8)
    
    for i in range(n_iters):
        # Move piece 1
        oc.batch_set_positions(move_src, piece_empty)
        oc.batch_set_positions(move_dst, piece_dst)
        
        # Get positions (simulate access pattern)
        # This will trigger rebuild because dirty
        pos_white = oc.get_positions(Color.WHITE)
        pos_black = oc.get_positions(Color.BLACK)
        
        # Verify result structure
        if len(pos_white) == 0:
            raise ValueError("Empty result")

        # Move back
        oc.batch_set_positions(move_dst, piece_empty)
        oc.batch_set_positions(move_src, piece_src)
        
        # Get again
        pos_white = oc.get_positions(Color.WHITE)
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Total time for {n_iters} iterations: {duration:.4f}s")
    print(f"Time per iteration: {duration/n_iters*1000:.4f}ms")
    print(f"Calls per second: {n_iters/duration:.2f}")

if __name__ == "__main__":
    benchmark_get_positions()
