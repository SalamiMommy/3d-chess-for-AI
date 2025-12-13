
import numpy as np
import sys
import os

# Ensure we can import from game3d
sys.path.append(os.getcwd())

from game3d.cache.manager import get_cache_manager
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.core.buffer import create_empty_buffer

# Mock board
class MockBoard:
    def get_initial_setup(self):
        return (np.empty((0, 3)), np.empty(0), np.empty(0))
    def __init__(self):
        self.generation = 0

def repro_attack_matrix():
    board = MockBoard()
    cache_mgr = get_cache_manager(board)
    move_cache = cache_mgr.move_cache
    
    # Create some dummy moves: White Rook at (0,0,0) attacking (0,0,1)...(0,0,5)
    # Move format: fx, fy, fz, tx, ty, tz
    moves = np.zeros((5, 6), dtype=COORD_DTYPE)
    for i in range(5):
        moves[i] = [0, 0, 0, 0, 0, i+1]
        
    print("Storing pseudolegal moves...")
    move_cache.store_pseudolegal_moves(Color.WHITE, moves)
    
    # Check if attack matrix is populated
    # Key for (0,0,0) piece
    piece_key = 0 # x=0, y=0, z=0
    
    # Check if (0,0,1) is targeted
    target_key = 0 | (0 << 9) | (1 << 18) # (0,0,1)
    
    # Target flat index
    target_flat = 0 + 0*9 + 1*81 # = 81
    
    # Check matrix
    # color=0 (White)
    blocks = move_cache._attack_matrix[0, target_flat]
    
    if np.any(blocks):
        print("SUCCESS: Attack matrix DOES contain data.")
    else:
        print("FAILURE: Attack matrix is EMPTY after store_pseudolegal_moves.")
        
    # Check calling store_piece_moves explicitly
    print("Explicitly calling store_piece_moves...")
    move_cache.store_piece_moves(Color.WHITE, piece_key, moves)
    
    blocks_after = move_cache._attack_matrix[0, target_flat]
    if np.any(blocks_after):
        print("CONFIRMED: store_piece_moves populates the matrix.")
    else:
        print("WEIRD: store_piece_moves didn't work?")

if __name__ == "__main__":
    repro_attack_matrix()
