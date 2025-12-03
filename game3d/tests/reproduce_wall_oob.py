
import sys
import os
import numpy as np

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock numba if not available
try:
    import numba
except ImportError:
    from unittest.mock import MagicMock
    numba = MagicMock()
    numba.njit = lambda func=None, **kwargs: (lambda *args, **kw: func(*args, **kw)) if func else (lambda f: f)
    numba.prange = range
    numba.int8 = int
    numba.int16 = int
    numba.int32 = int
    numba.int64 = int
    numba.float32 = float
    numba.float64 = float
    numba.boolean = bool
    sys.modules["numba"] = numba
    print("WARNING: Numba not found, using mock.")

from game3d.common.shared_types import PieceType, Color, SIZE
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.pieces.pieces.wall import generate_wall_moves

def reproduce_issue():
    print(f"Testing Wall bounds with SIZE={SIZE}")
    
    # Setup
    board = Board()
    cache = OptimizedCacheManager(board)
    
    # Place a wall at [0, 7, 6]
    # We suspect this generates a move to [0, 8, 6] which is invalid because
    # if the wall is 2x2, moving to y=8 means occupying y=8 and y=9 (OOB).
    start_pos = np.array([0, 7, 6], dtype=np.int16)
    
    print(f"Generating moves for Wall at {start_pos}")
    moves = generate_wall_moves(cache, Color.WHITE, start_pos)
    
    print(f"Generated {len(moves)} moves")
    
    invalid_move_target = np.array([0, 8, 6])
    found = False
    
    for i, m in enumerate(moves):
        # Move format: [from_x, from_y, from_z, to_x, to_y, to_z]
        dest = m[3:6]
        print(f"Move {i}: {m[:3]} -> {dest}")
        
        if np.array_equal(dest, invalid_move_target):
            found = True
            print("❌ FOUND INVALID MOVE TO [0, 8, 6]!")
            
    if found:
        print("SUCCESS: Reproduced the issue. The generator produced an OOB move.")
    else:
        print("FAILED: Did not reproduce the issue.")

if __name__ == "__main__":
    reproduce_issue()
