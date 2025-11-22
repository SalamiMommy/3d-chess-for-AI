
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE
from game3d.pieces.pieces.knight import generate_knight_moves
from game3d.pieces.pieces.speeder import generate_speeder_moves

def test_speeder_buff():
    print("Initializing GameState...")
    # Initialize empty game state with Board
    board = Board()
    game = GameState(board)
    
    # Clear board
    game.board.array().fill(0)
    # game.cache_manager.invalidate_all()
    
    # Setup positions
    # Speeder at (3, 3, 3)
    speeder_pos = np.array([3, 3, 3], dtype=COORD_DTYPE)
    
    # Buffed Knight at (3, 3, 4) - distance 1 from Speeder (inside aura)
    buffed_knight_pos = np.array([3, 3, 4], dtype=COORD_DTYPE)
    
    # Unbuffed Knight at (0, 0, 0) - far from Speeder
    unbuffed_knight_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    
    print("Placing pieces...")
    # Place pieces manually
    # Note: We need to update cache manager manually or via game methods if available
    # GameState.make_move updates caches, but we are setting up initial state.
    # We can use cache_manager.occupancy_cache.update directly if needed, 
    # but GameState usually handles this on init if passed a board, or we can just set the board and rebuild caches?
    # Let's try setting board and calling a refresh/rebuild if it exists, or just manually updating caches.
    
    # Actually, GameState initializes caches from board.
    # So we should set board and then re-initialize or manually update.
    # Let's just manually update the occupancy cache.
    
    # Set Speeder
    game.board[3, 3, 3] = PieceType.SPEEDER | Color.WHITE
    game.cache_manager.occupancy_cache.on_occupancy_changed(
        np.array([[3, 3, 3]], dtype=COORD_DTYPE), 
        np.array([[PieceType.SPEEDER | Color.WHITE]], dtype=np.int8)
    )
    
    # Set Buffed Knight
    game.board[3, 3, 4] = PieceType.KNIGHT | Color.WHITE
    game.cache_manager.occupancy_cache.on_occupancy_changed(
        np.array([[3, 3, 4]], dtype=COORD_DTYPE), 
        np.array([[PieceType.KNIGHT | Color.WHITE]], dtype=np.int8)
    )
    
    # Set Unbuffed Knight
    game.board[0, 0, 0] = PieceType.KNIGHT | Color.WHITE
    game.cache_manager.occupancy_cache.on_occupancy_changed(
        np.array([[0, 0, 0]], dtype=COORD_DTYPE), 
        np.array([[PieceType.KNIGHT | Color.WHITE]], dtype=np.int8)
    )
    
    # Ensure aura cache is updated
    # Aura cache listens to occupancy changes, so it should be updated automatically via cache manager dispatch
    # But we need to make sure cache_manager.occupancy_cache dispatches events.
    # Looking at auracache.py, it implements CacheListener.
    # We might need to trigger a full refresh or ensure listeners are hooked up.
    # GameState init hooks them up.
    
    # Let's verify aura cache state
    aura_cache = None
    for cache in game.cache_manager._effect_cache_instances:
        if cache.__class__.__name__ == 'ConsolidatedAuraCache':
            aura_cache = cache
            break
            
    if aura_cache:
        print("Checking aura cache...")
        is_buffed = aura_cache.batch_is_buffed(buffed_knight_pos.reshape(1, 3), Color.WHITE)[0]
        print(f"Knight at {buffed_knight_pos} buffed? {is_buffed}")
        
        is_unbuffed = aura_cache.batch_is_buffed(unbuffed_knight_pos.reshape(1, 3), Color.WHITE)[0]
        print(f"Knight at {unbuffed_knight_pos} buffed? {is_unbuffed}")
        
        if not is_buffed:
            print("ERROR: Knight should be buffed but isn't. Force updating aura cache.")
            # Try to force update if needed, but it should work if occupancy cache notified listeners
            # The on_occupancy_changed we called above might not have dispatched to listeners if we called it on the cache directly
            # depending on implementation.
            # Let's assume we need to manually notify aura cache for this test setup
            aura_cache.on_batch_occupancy_changed(
                np.array([[3, 3, 3], [3, 3, 4], [0, 0, 0]], dtype=COORD_DTYPE),
                np.array([[PieceType.SPEEDER], [PieceType.KNIGHT], [PieceType.KNIGHT]], dtype=np.int8) # Simplified types for aura cache?
                # Aura cache expects full piece values usually or handles extraction
            )
            # Re-check
            is_buffed = aura_cache.batch_is_buffed(buffed_knight_pos.reshape(1, 3), Color.WHITE)[0]
            print(f"Knight at {buffed_knight_pos} buffed after update? {is_buffed}")

    print("\nGenerating moves for Buffed Knight...")
    moves_buffed = generate_knight_moves(game.cache_manager, Color.WHITE, buffed_knight_pos)
    
    print(f"Generated {len(moves_buffed)} moves.")
    
    # Standard Knight moves: (1, 2, 0) etc.
    # Buffed: Longest component +1.
    # Example: (1, 2, 0) -> max=2. y becomes 3. -> (1, 3, 0)
    # Example: (2, 1, 0) -> max=2. x becomes 3. -> (3, 1, 0)
    
    # Let's check for a specific expected move
    # From (3, 3, 4):
    # Standard move (+1, +2, 0) -> (4, 5, 4)
    # Buffed move (+1, +3, 0) -> (4, 6, 4)
    
    expected_target = np.array([4, 6, 4], dtype=COORD_DTYPE)
    found = False
    for move in moves_buffed:
        # move is (from_x, from_y, from_z, to_x, to_y, to_z)
        target = move[3:6]
        if np.array_equal(target, expected_target):
            found = True
            break
            
    if found:
        print("SUCCESS: Found expected buffed move (4, 6, 4).")
    else:
        print("FAILURE: Did not find expected buffed move (4, 6, 4).")
        print("First 10 moves:")
        for i in range(min(10, len(moves_buffed))):
            print(moves_buffed[i])

    print("\nGenerating moves for Unbuffed Knight...")
    moves_unbuffed = generate_knight_moves(game.cache_manager, Color.WHITE, unbuffed_knight_pos)
    
    # From (0, 0, 0):
    # Standard move (+1, +2, 0) -> (1, 2, 0)
    # Buffed move would be (1, 3, 0)
    
    unexpected_target = np.array([1, 3, 0], dtype=COORD_DTYPE)
    found_unexpected = False
    for move in moves_unbuffed:
        target = move[3:6]
        if np.array_equal(target, unexpected_target):
            found_unexpected = True
            break
            
    if not found_unexpected:
        print("SUCCESS: Did not find buffed move for unbuffed knight.")
    else:
        print("FAILURE: Found buffed move for unbuffed knight!")

    # Check standard move exists for unbuffed
    expected_std = np.array([1, 2, 0], dtype=COORD_DTYPE)
    found_std = False
    for move in moves_unbuffed:
        target = move[3:6]
        if np.array_equal(target, expected_std):
            found_std = True
            break
            
    if found_std:
        print("SUCCESS: Found standard move (1, 2, 0) for unbuffed knight.")
    else:
        print("FAILURE: Did not find standard move (1, 2, 0) for unbuffed knight.")

if __name__ == "__main__":
    test_speeder_buff()
