import numpy as np
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE, PIECE_TYPE_DTYPE, SIZE
from game3d.pieces.pieces.wall import generate_wall_moves

def verify_wall_logic():
    print(f"Testing Wall logic with SIZE={SIZE}")
    
    # Setup
    board = Board()
    cache = OptimizedCacheManager(board)
    cache.occupancy_cache.clear()
    
    # Place Wall at [3, 6, 6]
    start_pos = np.array([3, 6, 6], dtype=COORD_DTYPE)
    wall_piece = np.array([PieceType.WALL, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
    
    # Populate 2x2 block
    block_offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=COORD_DTYPE)
    for offset in block_offsets:
        pos = start_pos + offset
        cache.occupancy_cache.set_position(pos, wall_piece)
        
    print(f"Wall placed at {start_pos}")
    
    # Generate moves
    moves = generate_wall_moves(cache, Color.WHITE, start_pos)
    
    print(f"Generated {len(moves)} moves")
    
    # Check for [8, 0, 3]
    target = np.array([8, 0, 3], dtype=COORD_DTYPE)
    found = False
    for mv in moves:
        to_pos = mv[3:]
        if np.array_equal(to_pos, target):
            found = True
            print(f"FOUND INVALID MOVE: {mv}")
            break
            
    if not found:
        print("Invalid move [8, 0, 3] was NOT generated.")
        
    # Check for any OOB moves
    for mv in moves:
        to_x, to_y, to_z = mv[3], mv[4], mv[5]
        if to_x >= SIZE - 1 or to_y >= SIZE - 1:
            print(f"FOUND OOB MOVE: {mv}")

if __name__ == "__main__":
    verify_wall_logic()
