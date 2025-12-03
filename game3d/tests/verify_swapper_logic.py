


import numpy as np
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE
from game3d.pieces.pieces.swapper import generate_swapper_moves

def verify_swapper():
    print("Verifying Swapper Logic...")
    
    # 1. Setup Board
    game = GameState.from_startpos()
    game.cache_manager.occupancy_cache.clear()
    
    # Place Swapper at [4, 4, 4]
    swapper_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    game.cache_manager.occupancy_cache.set_position(swapper_pos, np.array([PieceType.SWAPPER.value, Color.WHITE.value]))
    
    # Place Friendly Pawn at [4, 4, 5] (adjacent)
    friendly_pos = np.array([4, 4, 5], dtype=COORD_DTYPE)
    game.cache_manager.occupancy_cache.set_position(friendly_pos, np.array([PieceType.PAWN.value, Color.WHITE.value]))
    
    # Place Enemy Pawn at [4, 5, 4] (adjacent)
    enemy_pos = np.array([4, 5, 4], dtype=COORD_DTYPE)
    game.cache_manager.occupancy_cache.set_position(enemy_pos, np.array([PieceType.PAWN.value, Color.BLACK.value]))
    
    # 2. Verify Move Generation
    print("\n--- Move Generation ---")
    moves = generate_swapper_moves(game.cache_manager, Color.WHITE, swapper_pos)
    
    # Check for King moves (adjacent empty squares)
    # Expect moves to all adjacent squares except occupied ones (unless capture/swap allowed)
    # Swapper should be able to move to friendly (swap) and enemy (capture)
    
    has_swap_move = False
    has_capture_move = False
    
    for move in moves:
        dest = move[3:6]
        if np.array_equal(dest, friendly_pos):
            has_swap_move = True
            print(f"Found swap move to {dest}")
        if np.array_equal(dest, enemy_pos):
            has_capture_move = True
            print(f"Found capture move to {dest}")
            
    if not has_swap_move:
        print("FAIL: No move generated to friendly piece (Swap missing)")
    else:
        print("PASS: Move generated to friendly piece")
        
    if not has_capture_move:
        print("FAIL: No move generated to enemy piece (King capture missing)")
    else:
        print("PASS: Move generated to enemy piece")

    # 3. Verify Move Execution (Swap)
    print("\n--- Move Execution (Swap) ---")
    if has_swap_move:
        # Construct the move
        swap_move = np.concatenate([swapper_pos, friendly_pos]).astype(COORD_DTYPE)
        
        try:
            # Execute move
            new_state = game.make_move_vectorized(swap_move)
            
            # Check positions
            piece_at_swapper_start = new_state.cache_manager.occupancy_cache.get(swapper_pos)
            piece_at_friendly_start = new_state.cache_manager.occupancy_cache.get(friendly_pos)
            
            print(f"Piece at old Swapper pos {swapper_pos}: {piece_at_swapper_start}")
            print(f"Piece at old Friendly pos {friendly_pos}: {piece_at_friendly_start}")
            
            # Expect:
            # Old Swapper pos -> Friendly Pawn
            # Old Friendly pos -> Swapper
            
            is_swapped = False
            if (piece_at_swapper_start is not None and 
                piece_at_swapper_start['piece_type'] == PieceType.PAWN.value and
                piece_at_friendly_start is not None and
                piece_at_friendly_start['piece_type'] == PieceType.SWAPPER.value):
                is_swapped = True
                
            if is_swapped:
                print("PASS: Pieces successfully swapped!")
            else:
                print("FAIL: Pieces did NOT swap correctly.")
                if piece_at_swapper_start is None and piece_at_friendly_start['piece_type'] == PieceType.SWAPPER.value:
                     print("      Result looks like a capture (friendly piece removed).")
        except Exception as e:
            print(f"FAIL: Move execution raised exception: {e}")

if __name__ == "__main__":
    verify_swapper()
