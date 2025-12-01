
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.common.shared_types import PieceType, Color, SIZE
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.game.gamestate import GameState
from game3d.pieces.pieces.wall import generate_wall_moves

def test_wall_bounds():
    print(f"Testing Wall bounds with SIZE={SIZE}")
    
    # Setup
    board = Board()
    cache = OptimizedCacheManager(board)
    state = GameState(board=board, cache_manager=cache, color=Color.WHITE)
    
    # Place a wall at [5, 7, 4]
    # This is valid: occupies y=7, y=8.
    start_pos = np.array([5, 7, 4], dtype=np.int16)
    
    # Manually set occupancy for the wall (4 squares)
    # We don't need full game state setup, just enough for generate_wall_moves
    # generate_wall_moves uses cache_manager.occupancy_cache._occ
    
    # But wait, generate_wall_moves takes (cache_manager, color, pos)
    # It checks occupancy of TARGET squares.
    # It does NOT check if start_pos is valid (it assumes it is).
    
    moves = generate_wall_moves(cache, Color.WHITE, start_pos)
    
    print(f"Generated {len(moves)} moves for Wall at {start_pos}")
    for i in range(len(moves)):
        m = moves[i]
        dest = m[3:6]
        print(f"Move {i}: {m[:3]} -> {dest}")
        
        if dest[1] == 8:
            print("❌ FOUND INVALID MOVE TO y=8!")
            
    # Check specifically for [5, 8, 4]
    invalid_move = np.array([5, 8, 4])
    found = False
    for m in moves:
        if np.array_equal(m[3:6], invalid_move):
            found = True
            break
            
    if found:
        print("FAILURE: Generated invalid move to [5, 8, 4]")
    else:
        print("SUCCESS: Did not generate invalid move to [5, 8, 4]")

if __name__ == "__main__":
    test_wall_bounds()
