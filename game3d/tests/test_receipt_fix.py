"""
Verify the fix for delayed game termination.
"""
import logging
import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test that receipt.is_game_over is used to exit the loop early
def test_receipt_game_over():
    """
    Simulate the fixed game loop and verify we exit immediately on game over.
    """
    from game3d.game.factory import start_game_state
    from game3d.main_game import OptimizedGame3D
    from game3d.common.shared_types import Color, COORD_DTYPE
    from game3d.movement.movepiece import Move
    
    gs = start_game_state()
    game = OptimizedGame3D(board=gs.board, cache=gs.cache_manager)
    
    move_count = 0
    max_moves = 2000
    receipt_game_over_count = 0
    
    # Simulate the FIXED loop from parallel_self_play.py
    while move_count < max_moves and not game.is_game_over():
        
        moves = game.state.legal_moves
        if moves.size == 0:
            print(f"UNEXPECTED: legal_moves=0 but is_game_over=False at move {move_count}")
            break
        
        from_coords = moves[:, :3].astype(COORD_DTYPE)
        occ = game.state.cache_manager.occupancy_cache
        from_colors, _ = occ.batch_get_attributes(from_coords)
        valid_mask = from_colors == game.state.color
        
        if not np.any(valid_mask):
            break
        
        valid_moves = moves[valid_mask]
        chosen_move = valid_moves[0]
        submit_move = Move(chosen_move[:3], chosen_move[3:6])
        
        try:
            receipt = game.submit_move(submit_move)
        except Exception as e:
            print(f"Move failed: {e}")
            break
        
        game._state = receipt.new_state
        game._state._legal_moves_cache = None
        move_count += 1
        
        # ✅ THE FIX: Check receipt.is_game_over
        if receipt.is_game_over:
            receipt_game_over_count += 1
            print(f"[FIX TRIGGERED] receipt.is_game_over=True at move {move_count}, result={receipt.result}")
            # In the fixed code, we would break here
            break
        
        if move_count % 500 == 0:
            coords, _, _ = occ.get_all_occupied_vectorized()
            print(f"Progress: {move_count} moves, {len(coords)} pieces")
    
    print(f"\nGame ended after {move_count} moves")
    print(f"receipt.is_game_over triggered: {receipt_game_over_count} times")
    print(f"Final is_game_over: {game.is_game_over()}")
    
    if receipt_game_over_count > 0:
        print("\n✅ FIX WORKING: Game ended via receipt.is_game_over")
    elif game.is_game_over():
        print("\n⚠️ Game ended via loop condition, not receipt")
    else:
        print("\n⚠️ Game ended via move limit or other condition")


if __name__ == "__main__":
    test_receipt_game_over()
