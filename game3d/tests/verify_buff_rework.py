
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color, COORD_DTYPE
from game3d.pieces.pieces.kinglike import generate_king_moves, KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS
from game3d.pieces.pieces.panel import generate_panel_moves, PANEL_MOVEMENT_VECTORS, BUFFED_PANEL_MOVEMENT_VECTORS
from game3d.pieces.pieces.orbiter import generate_orbital_moves, _ORBITAL_DIRS, _BUFFED_ORBITAL_DIRS

def verify_moves(generated_moves, expected_vectors, start_pos):
    """Verify that generated moves match expected vectors from a start position."""
    if len(generated_moves) == 0:
        if len(expected_vectors) == 0:
            return True
        print(f"Expected {len(expected_vectors)} moves, got 0")
        return False
        
    # Extract targets
    targets = generated_moves[:, 3:6]
    
    # Calculate actual vectors
    actual_vectors = targets - start_pos
    
    # Sort for comparison
    # Use lexsort
    expected_sorted = expected_vectors[np.lexsort((expected_vectors[:, 2], expected_vectors[:, 1], expected_vectors[:, 0]))]
    actual_sorted = actual_vectors[np.lexsort((actual_vectors[:, 2], actual_vectors[:, 1], actual_vectors[:, 0]))]
    
    # Filter out out-of-bounds expected vectors if any (though we usually test in center)
    # But here we assume start_pos is central enough
    
    if expected_sorted.shape != actual_sorted.shape:
        print(f"Shape mismatch: Expected {expected_sorted.shape}, Got {actual_sorted.shape}")
        # Print differences
        # set of tuples
        exp_set = set(map(tuple, expected_sorted))
        act_set = set(map(tuple, actual_sorted))
        print("Missing:", exp_set - act_set)
        print("Extra:", act_set - exp_set)
        return False
        
    if not np.array_equal(expected_sorted, actual_sorted):
        print("Content mismatch")
        return False
        
    return True

def main():
    print("Verifying Buff Rework...")
    
    # Setup
    game = GameState.from_startpos()
    # Clear board
    game.cache_manager.occupancy_cache._occ.fill(0)
    
    # Center position
    center = np.array([4, 4, 4], dtype=COORD_DTYPE)
    
    # 1. Test King
    print("\nTesting King...")
    # Unbuffed
    game.cache_manager.consolidated_aura_cache._buffed_squares.fill(False)
    moves = generate_king_moves(game.cache_manager, Color.WHITE, center, PieceType.KING)
    if verify_moves(moves, KING_MOVEMENT_VECTORS, center):
        print("King Unbuffed: PASS")
    else:
        print("King Unbuffed: FAIL")
        
    # Buffed
    game.cache_manager.consolidated_aura_cache._buffed_squares[4, 4, 4] = True
    moves = generate_king_moves(game.cache_manager, Color.WHITE, center, PieceType.KING)
    if verify_moves(moves, BUFFED_KING_MOVEMENT_VECTORS, center):
        print("King Buffed: PASS")
    else:
        print("King Buffed: FAIL")
        
    # 2. Test Panel
    print("\nTesting Panel...")
    # Unbuffed
    game.cache_manager.consolidated_aura_cache._buffed_squares.fill(False)
    moves = generate_panel_moves(game.cache_manager, Color.WHITE, center)
    if verify_moves(moves, PANEL_MOVEMENT_VECTORS, center):
        print("Panel Unbuffed: PASS")
    else:
        print("Panel Unbuffed: FAIL")
        
    # Buffed
    game.cache_manager.consolidated_aura_cache._buffed_squares[4, 4, 4] = True
    moves = generate_panel_moves(game.cache_manager, Color.WHITE, center)
    if verify_moves(moves, BUFFED_PANEL_MOVEMENT_VECTORS, center):
        print("Panel Buffed: PASS")
    else:
        print("Panel Buffed: FAIL")

    # 3. Test Orbiter
    print("\nTesting Orbiter...")
    # Unbuffed
    game.cache_manager.consolidated_aura_cache._buffed_squares.fill(False)
    moves = generate_orbital_moves(game.cache_manager, Color.WHITE, center)
    if verify_moves(moves, _ORBITAL_DIRS, center):
        print("Orbiter Unbuffed: PASS")
    else:
        print("Orbiter Unbuffed: FAIL")
        
    # Buffed
    game.cache_manager.consolidated_aura_cache._buffed_squares[4, 4, 4] = True
    moves = generate_orbital_moves(game.cache_manager, Color.WHITE, center)
    if verify_moves(moves, _BUFFED_ORBITAL_DIRS, center):
        print("Orbiter Buffed: PASS")
    else:
        print("Orbiter Buffed: FAIL")

if __name__ == "__main__":
    main()
