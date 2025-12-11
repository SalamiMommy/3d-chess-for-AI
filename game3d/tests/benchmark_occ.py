
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.cache.caches.occupancycache import OccupancyCache
from game3d.common.shared_types import SIZE, PieceType, Color

def run_benchmark():
    print(f"Benchmarking OccupancyCache (SIZE={SIZE})...")
    
    cache = OccupancyCache(board_size=SIZE)
    
    # Setup initial state
    # Randomly place ~32 pieces
    np.random.seed(42)
    
    n_pieces = 32
    coords = []
    pieces = []
    
    for _ in range(n_pieces):
        while True:
            c = np.random.randint(0, SIZE, size=(3,))
            if not cache.is_occupied_at(c[0], c[1], c[2]):
                coords.append(c)
                break
        
        ptype = np.random.randint(1, 7)
        color = Color.WHITE if np.random.random() > 0.5 else Color.BLACK
        pieces.append([ptype, color])

    coords = np.array(coords, dtype=np.int16)
    pieces = np.array(pieces, dtype=np.int8) # Assuming int8 for piece/color
    
    # Use rebuild or batch_set_positions
    cache.rebuild(coords, pieces[:, 0], pieces[:, 1])
    
    print("Initial state set.")
    
    iterations = 10000
    start_time = time.perf_counter()
    
    # Simulate move loop
    # In each iteration: pick a piece, move it to empty square
    # Then call get_positions
    
    # We'll just toggle a square for simplicity to avoid tracking valid moves
    # Toggle (0,0,0) between occupied and empty
    
    coord = np.array([0, 0, 0], dtype=np.int16).reshape(1, 3)
    piece_white = np.array([PieceType.PAWN, Color.WHITE], dtype=np.int8) # Fix: 1D array
    piece_empty = None # Represents clearing
    
    active_piece = piece_white
    
    for i in range(iterations):
        # Update board
        if i % 2 == 0:
            cache.set_position(coord, active_piece)
        else:
            cache.set_position(coord, None)
            
        # Access positions (critical path)
        _ = cache.get_positions(Color.WHITE)
        _ = cache.get_positions(Color.BLACK)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    ops_per_sec = iterations / duration
    
    print(f"Total time: {duration:.4f}s")
    print(f"Iterations: {iterations}")
    print(f"Ops/sec: {ops_per_sec:.2f}")

if __name__ == "__main__":
    run_benchmark()
