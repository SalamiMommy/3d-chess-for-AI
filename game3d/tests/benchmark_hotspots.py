
import time
import numpy as np
import torch
from game3d.game.gamestate import GameState
from game3d.pieces.pieces.wall import generate_wall_moves
from game3d.pieces.pieces.friendlytp import generate_friendlytp_moves
from game3d.pieces.pieces.infiltrator import _get_valid_pawn_targets
from game3d.attacks.check import _square_attacked_by_slow
from game3d.common.shared_types import Color, PieceType, SIZE, COORD_DTYPE
from game3d.cache.manager import get_cache_manager
from game3d.board.board import Board

def benchmark():
    print("Setting up benchmark...")
    
    # Initialize empty board and cache manager
    board = Board()
    cache_manager = get_cache_manager(board)
    
    # Helper to add piece
    def add_piece(pos, ptype, color):
        coord = np.array(pos, dtype=COORD_DTYPE)
        piece_data = np.array([ptype, color], dtype=np.int32)
        cache_manager.occupancy_cache.set_position(coord, piece_data)

    # Populate board
    # Add some Walls
    for x in range(0, 8, 2):
        for y in range(0, 8, 2):
            add_piece([x, y, 0], PieceType.WALL, Color.WHITE)
            
    # Add FriendlyTP and friends
    add_piece([4, 4, 4], PieceType.FRIENDLYTELEPORTER, Color.WHITE)
    for i in range(20):
        add_piece([i%9, (i+1)%9, 5], PieceType.PAWN, Color.WHITE)
        
    # Add Infiltrator and Enemy Pawns
    add_piece([5, 5, 5], PieceType.INFILTRATOR, Color.WHITE)
    for i in range(20):
        add_piece([i%9, (i+2)%9, 6], PieceType.PAWN, Color.BLACK)
        
    # Add pieces for check detection
    add_piece([4, 4, 8], PieceType.KING, Color.WHITE)
    # Add many attackers
    for i in range(10):
        add_piece([i%9, 0, 8], PieceType.ROOK, Color.BLACK)
        add_piece([0, i%9, 8], PieceType.QUEEN, Color.BLACK)

    # Create GameState (needed for some functions that expect it)
    state = GameState(board, Color.WHITE, cache_manager)
    
    print("Starting benchmarks...")
    
    # 1. generate_wall_moves
    start = time.time()
    for _ in range(100):
        # Find a wall
        wall_pos = np.array([0, 0, 0])
        generate_wall_moves(cache_manager, Color.WHITE.value, wall_pos)
    print(f"generate_wall_moves (100 iters): {time.time() - start:.4f}s")
    
    # 2. generate_friendlytp_moves
    start = time.time()
    ftp_pos = np.array([4, 4, 4])
    for _ in range(100):
        generate_friendlytp_moves(cache_manager, Color.WHITE.value, ftp_pos)
    print(f"generate_friendlytp_moves (100 iters): {time.time() - start:.4f}s")
    
    # 3. _get_valid_pawn_targets
    start = time.time()
    for _ in range(1000):
        _get_valid_pawn_targets(cache_manager, Color.WHITE.value, teleport_behind=False)
    print(f"_get_valid_pawn_targets (1000 iters): {time.time() - start:.4f}s")
    
    # 4. square_attacked_by_fast
    # This is the big one.
    from game3d.attacks.fast_attack import square_attacked_by_fast
    target_sq = np.array([4, 4, 8]) # King pos
    start = time.time()
    for _ in range(100): # More iters now that it's fast
        square_attacked_by_fast(board, target_sq, Color.BLACK.value, cache_manager)
    print(f"square_attacked_by_fast (100 iters): {time.time() - start:.4f}s")

if __name__ == "__main__":
    benchmark()
