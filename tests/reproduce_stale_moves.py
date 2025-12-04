import numpy as np
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.game3d import OptimizedGame3D
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE, PIECE_TYPE_DTYPE, SIZE
from game3d.movement.movepiece import Move

def reproduce_stale_moves():
    print("Reproducing Stale Moves Scenario...")
    
    # Setup
    board = Board()
    cache = OptimizedCacheManager(board)
    game = OptimizedGame3D(board=board, cache=cache)
    cache.occupancy_cache.clear()
    
    # 1. Place Swapper at [3, 6, 6]
    start_pos = np.array([3, 6, 6], dtype=COORD_DTYPE)
    swapper_piece = np.array([PieceType.SWAPPER, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
    cache.occupancy_cache.set_position(start_pos, swapper_piece)
    
    # Place a Friendly piece at [8, 0, 3] (Target for swap)
    # Note: [8, 0, 3] is valid for Swapper but invalid for Wall
    target_pos = np.array([8, 0, 3], dtype=COORD_DTYPE)
    friendly_piece = np.array([PieceType.PAWN, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
    cache.occupancy_cache.set_position(target_pos, friendly_piece)
    
    # Add a King to prevent generator from filtering all moves
    king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    king_piece = np.array([PieceType.KING, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
    cache.occupancy_cache.set_position(king_pos, king_piece)
    
    print(f"Placed Swapper at {start_pos} and Friendly Pawn at {target_pos}")
    
    # Debug: Check friendly positions
    friendly_pos = cache.occupancy_cache.get_positions(Color.WHITE)
    print(f"Friendly positions: {friendly_pos}")
    
    # 2. Generate Legal Moves (Should include Swap to [8, 0, 3])
    moves = game.state.legal_moves
    print(f"Generated {len(moves)} moves")
    
    swap_move = None
    for mv in moves:
        if np.array_equal(mv[:3], start_pos) and np.array_equal(mv[3:6], target_pos):
            swap_move = mv
            break
            
    if swap_move is None:
        print("ERROR: Swapper did not generate swap move to [8, 0, 3]")
        return
        
    print(f"Found Swap Move: {swap_move}")
    
    # 3. CHANGE THE PIECE to a Wall (Simulate state desync)
    print("Changing piece at [3, 6, 6] to Wall...")
    wall_piece = np.array([PieceType.WALL, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
    cache.occupancy_cache.set_position(start_pos, wall_piece)
    
    # Also need to clear the target so it looks like a move to empty? 
    # Or keep it? Wall can't move to friendly.
    # But let's see what validation says.
    
    # 4. Submit the STALE Swapper move
    print("Submitting Stale Swap Move...")
    move_obj = Move(swap_move[:3], swap_move[3:6])
    
    try:
        game.submit_move(move_obj)
        print("Move submitted successfully (Unexpected!)")
    except Exception as e:
        print(f"Caught expected exception: {e}")
        print("Simulating fix: Invalidating cache and regenerating moves...")
        
        # Simulate the fix
        game.state.cache_manager.move_cache.invalidate()
        game.state._legal_moves_cache = None
        
        # Debug: Verify piece type in cache
        piece_info = game.state.cache_manager.occupancy_cache.get(start_pos)
        ptype = piece_info["piece_type"] if piece_info else "None"
        print(f"Piece type at {start_pos} after change: {ptype} (Expected: {PieceType.WALL})")
        
        # Regenerate moves
        new_moves = game.state.legal_moves
        print(f"Regenerated {len(new_moves)} moves")
        
        # Check if the invalid swap move is still there
        swap_move_present = False
        for mv in new_moves:
            if np.array_equal(mv[:3], start_pos) and np.array_equal(mv[3:6], target_pos):
                swap_move_present = True
                break
        
        if not swap_move_present:
            print("SUCCESS: Invalid swap move is GONE after cache invalidation!")
        else:
            print("FAILURE: Invalid swap move PERSISTS after cache invalidation!")

if __name__ == "__main__":
    reproduce_stale_moves()
