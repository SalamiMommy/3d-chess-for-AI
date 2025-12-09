
import sys
import os
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE
from game3d.movement.movepiece import Move
from game3d.game.turnmove import make_move

def idx(x, y, z):
    return np.array([x, y, z], dtype=COORD_DTYPE)

def verify_invalidation():
    print("Initializing components...")
    board = Board()
    # Cache Manager will be created by GameState or manually
    
    # Setup Board:
    # White Rook at (0, 0, 0)
    # White Pawn at (0, 4, 0) (Blocking)
    # Black King at (0, 8, 0) (Target)
    
    coords = np.array([
        [0, 0, 0],
        [0, 4, 0],
        [0, 8, 0]
    ], dtype=COORD_DTYPE)
    
    pieces = np.array([
        [PieceType.ROOK, Color.WHITE],
        [PieceType.KNIGHT, Color.WHITE], # Changed to Knight to allow easy unblock
        [PieceType.KING, Color.BLACK],
        [PieceType.KING, Color.WHITE]
    ], dtype=np.int8) # Using int8 matching standard DTYPE usually
    
    # Manually init manager to inject setup
    # Coordinates for the 4 pieces
    coords_full = np.vstack([coords, [7, 7, 7]]) # White King at 7,7,7
    
    manager = OptimizedCacheManager(board)
    manager._initialize_from_setup((coords_full, pieces[:, 0], pieces[:, 1]))
    
    game_state = GameState(board, Color.WHITE, manager)
    
    # Initial Check
    print("Checking initial state (Blocked)...")
    
    # Ensure moves are generated
    moves_white = manager.move_cache.get_pseudolegal_moves(Color.WHITE.value)
    if moves_white is None:
        # Force generation using generator to populate cache properly
        import game3d.movement.generator as gen_module
        if gen_module._generator is None:
            gen_module.initialize_generator()
        gen_module._generator.refresh_pseudolegal_moves(game_state)
        moves_white = manager.move_cache.get_pseudolegal_moves(Color.WHITE.value)

    if moves_white is None:
        print("FAIL: Could not generate moves")
        return

    # Verify Rook does NOT attack King
    king_pos = np.array([0, 8, 0])
    attacks_king = False
    
    # Find moves for Rook
    # We can check the move cache content
    for move in moves_white:
        if np.array_equal(move[3:6], king_pos):
            attacks_king = True
            break
            
    if attacks_king:
        print("FAIL: Rook attacks King despite blocker!")
        return
    else:
        print("PASS: Rook correctly blocked.")
        
    # --- ACTION: Move Knight to (2, 5, 0) (Unblock) ---
    print("\nMoving Knight to (2, 5, 0) (Unblock)...")
    move = np.array([0, 4, 0, 2, 5, 0], dtype=np.int8) # Valid Knight Jump
    
    # Use standard make_move to trigger detailed invalidation logic
    new_state = make_move(game_state, move)
    
    # Note: make_move returns new state with OPPONENT color (Black)
    # But we want to check WHITE's moves (the previous player)
    # White's moves should have been invalidated and potentially regenerated (or marked for regen).
    
    print("Checking cache state after move...")
    
    # Check if White's moves are now valid/regenerated correctly
    # Access cache from manager (shared)
    
    # Check if cache thinks it needs regeneration or has new moves
    # We request moves for WHITE. 
    # Since current state is BLACK, we need to temporarily query White
    
    # Force regeneration if needed via generator
    # In `make_move`, we added logic to `refresh_pseudolegal_moves` for the just-moved player.
    # So `moves_white` should be fresh!
    
    moves_white_new = manager.move_cache.get_pseudolegal_moves(Color.WHITE.value)
    
    if moves_white_new is None:
        print("Cache is None (Correctly Invalidated/Waiting for Regen)")
        # Force regen to check content
        from game3d.movement.generator import _get_generator
        gen = _get_generator()
        # We need a state with White as active to generate white moves??
        # Generator usually takes state.color
        dummy_state = GameState(new_state.board, Color.WHITE, manager)
        moves_white_new = gen.generate_pseudolegal_moves_from_scratch(dummy_state)
    
    # Verify Rook NOW attacks King
    attacks_king_new = False
    rook_attacks_count = 0
    
    if moves_white_new is not None:
         for move in moves_white_new:
            if np.array_equal(move[3:6], king_pos):
                attacks_king_new = True
            if np.array_equal(move[:3], [0,0,0]):
                rook_attacks_count += 1
                
    print(f"Rook moves found: {rook_attacks_count}")
    
    if attacks_king_new:
        print("PASS: Rook correctly attacks King after unblock.")
    else:
        print("FAIL: Rook DOES NOT attack King after unblock! (Cache Desync)")
        print("This reproduces the 'Attackers: Unknown' issue where cache ignores the new attack line.")
        return

    
    # --- INTERMEDIATE: Black makes a dummy move to pass turn back to White ---
    # Black King at (0, 8, 0). Move to (1, 8, 0).
    print("Black making dummy move (K 0,8,0 -> 1,8,0)...")
    move_black = np.array([0, 8, 0, 1, 8, 0], dtype=np.int8)
    
    # Validate black move first just in case
    # GameState is new_state (Black to move)
    state_after_black = make_move(new_state, move_black)

    # --- ACTION: Move Knight BACK to (0, 4, 0) (BLOCK AGAIN) ---
    print("\nMoving Knight BACK to (0, 4, 0) (Block Again)...")
    # Knight is now at (2, 5, 0). Moving back to (0, 4, 0).
    move_back = np.array([2, 5, 0, 0, 4, 0], dtype=np.int8)
    
    new_state_2 = make_move(state_after_black, move_back)
    
    # Check if White's moves are now BLOCKED again
    # We query moves for WHITE again.
    
    # Force regen?
    moves_white_blocked = manager.move_cache.get_pseudolegal_moves(Color.WHITE.value)
    
    if moves_white_blocked is None:
         # Need to trigger regen
         gen_module._generator.refresh_pseudolegal_moves(new_state_2) # using new state info but technically we want White moves
         # Wait, refresh_pseudolegal_moves relies on state.color. new_state_2 is BLACK turn.
         # So we need a proxy for White again.
         dummy_state_white = GameState(new_state_2.board, Color.WHITE, manager)
         gen_module._generator.refresh_pseudolegal_moves(dummy_state_white)
         moves_white_blocked = manager.move_cache.get_pseudolegal_moves(Color.WHITE.value)
         
    attacks_king_blocked = False
    if moves_white_blocked is not None:
         for move in moves_white_blocked:
            if np.array_equal(move[3:6], king_pos):
                attacks_king_blocked = True
                break
    
    if attacks_king_blocked:
        print("FAIL: Rook STILL attacks King after blocking! (Ghost Attack / Stale Bitboard)")
    else:
        print("PASS: Rook correctly blocked again.")

if __name__ == "__main__":
    verify_invalidation()
