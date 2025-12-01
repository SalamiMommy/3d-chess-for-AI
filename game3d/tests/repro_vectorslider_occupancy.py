import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType
from game3d.pieces.pieces.vectorslider import generate_vector_slider_moves

def repro_vectorslider_occupancy():
    print("Reproducing VectorSlider occupancy issue...")
    
    # Initialize game state
    game = GameState.from_startpos()
    
    # Clear the board
    game.cache_manager.occupancy_cache.clear()
    
    # Test Case 1: Friendly Blocker with legal_moves
    print("\nTest Case 1: Friendly Blocker (via legal_moves)")
    
    # Place Kings (required for legal_moves)
    white_king_pos = np.array([0, 8, 8])
    white_king_piece = np.array([PieceType.KING, Color.WHITE])
    game.cache_manager.occupancy_cache.set_position(white_king_pos, white_king_piece)
    
    black_king_pos = np.array([8, 8, 8])
    black_king_piece = np.array([PieceType.KING, Color.BLACK])
    game.cache_manager.occupancy_cache.set_position(black_king_pos, black_king_piece)
    
    # Place a White VectorSlider at [0, 0, 0]
    vs_pos = np.array([0, 0, 0])
    vs_piece = np.array([PieceType.VECTORSLIDER, Color.WHITE])
    game.cache_manager.occupancy_cache.set_position(vs_pos, vs_piece)
    
    # Place a White Pawn at [1, 2, 0]
    blocker_pos = np.array([1, 2, 0])
    blocker_piece = np.array([PieceType.PAWN, Color.WHITE])
    game.cache_manager.occupancy_cache.set_position(blocker_pos, blocker_piece)
    
    # Generate legal moves
    moves = game.legal_moves
    
    # Filter moves for VectorSlider at [0, 0, 0]
    # moves is (N, 6) array: [fx, fy, fz, tx, ty, tz]
    vs_moves_mask = (moves[:, 0] == 0) & (moves[:, 1] == 0) & (moves[:, 2] == 0)
    vs_moves = moves[vs_moves_mask]
    
    print(f"Generated {len(vs_moves)} moves for VectorSlider.")
    
    move_destinations = [tuple(m[3:6]) for m in vs_moves]
    
    if tuple(blocker_pos) in move_destinations:
        print("FAIL: Friendly blocker at [1, 2, 0] was included in moves (should be blocked).")
        return True
        
    target_pos = np.array([2, 4, 0])
    if tuple(target_pos) in move_destinations:
        print("FAIL: Moved past friendly blocker to [2, 4, 0].")
        return True
        
    print("PASS: Friendly blocker worked correctly.")
    
    # Test Case 2: Enemy Blocker with legal_moves
    print("\nTest Case 2: Enemy Blocker (via legal_moves)")
    
    # Re-initialize game state for clean test
    game = GameState.from_startpos()
    game.cache_manager.occupancy_cache.clear()
    
    # Place Kings again
    game.cache_manager.occupancy_cache.set_position(white_king_pos, white_king_piece)
    game.cache_manager.occupancy_cache.set_position(black_king_pos, black_king_piece)
    
    # Place a White VectorSlider at [0, 0, 0]
    game.cache_manager.occupancy_cache.set_position(vs_pos, vs_piece)
    
    # Place a Black Pawn at [1, 2, 0]
    enemy_pos = np.array([1, 2, 0])
    enemy_piece = np.array([PieceType.PAWN, Color.BLACK])
    game.cache_manager.occupancy_cache.set_position(enemy_pos, enemy_piece)
    
    # Generate legal moves
    moves = game.legal_moves
    
    # Filter moves for VectorSlider
    vs_moves_mask = (moves[:, 0] == 0) & (moves[:, 1] == 0) & (moves[:, 2] == 0)
    vs_moves = moves[vs_moves_mask]
    
    move_destinations = [tuple(m[3:6]) for m in vs_moves]
    
    capture_failed = False
    move_beyond_failed = False
    
    if tuple(enemy_pos) not in move_destinations:
        print("FAIL: Enemy blocker at [1, 2, 0] was NOT included in moves (should be capture).")
        capture_failed = True
        
    target_pos = np.array([2, 4, 0])
    if tuple(target_pos) in move_destinations:
        print("FAIL: Moved past enemy blocker to [2, 4, 0].")
        move_beyond_failed = True
        
    if capture_failed or move_beyond_failed:
        return True
        
    print("PASS: Enemy blocker worked correctly.")
    
    return False

if __name__ == "__main__":
    if repro_vectorslider_occupancy():
        sys.exit(1) # Exit with error if issue reproduced
    else:
        sys.exit(0)
