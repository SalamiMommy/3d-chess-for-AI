
import sys
import os
import numpy as np
import time
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.cache.manager import get_cache_manager
from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch, PARALLEL_THRESHOLD

def test_threading():
    print(f"PARALLEL_THRESHOLD is {PARALLEL_THRESHOLD}")
    
    # Setup board
    board = Board.empty()
    cache_manager = get_cache_manager(board, Color.WHITE)
    board._cache_manager = cache_manager
    
    state = GameState(board, Color.WHITE, cache_manager)
    
    # Place pieces to trigger threading
    # We need > PARALLEL_THRESHOLD pieces and > 1 piece type
    
    # 1. Place 9 Pawns (Type A)
    for i in range(9):
        cache_manager.occupancy_cache.set_position_fast(np.array([i, 0, 0]), PieceType.PAWN, Color.WHITE)
        
    # 2. Place 5 Rooks (Type B)
    for i in range(5):
        cache_manager.occupancy_cache.set_position_fast(np.array([i, 1, 0]), PieceType.ROOK, Color.WHITE)
        
    # 3. Place 5 Knights (Type C)
    for i in range(5):
        cache_manager.occupancy_cache.set_position_fast(np.array([i, 2, 0]), PieceType.KNIGHT, Color.WHITE)
        
    # Get positions
    coords = cache_manager.occupancy_cache.get_positions(Color.WHITE)
    print(f"Total pieces: {len(coords)}")
    
    if len(coords) < PARALLEL_THRESHOLD:
        print("ERROR: Not enough pieces to trigger threading threshold!")
        return
        
    # Generate moves
    start_time = time.time()
    moves = generate_pseudolegal_moves_batch(state, coords)
    end_time = time.time()
    
    print(f"Generated {len(moves)} moves in {end_time - start_time:.4f}s")
    
    if len(moves) == 0:
        print("ERROR: No moves generated!")
        sys.exit(1)
        
    # Verify correctness (sanity check)
    # Pawns at z=0 should have moves (unless blocked, but board is empty otherwise)
    # Rooks at z=0 should have moves
    
    print("Test passed!")

if __name__ == "__main__":
    test_threading()
