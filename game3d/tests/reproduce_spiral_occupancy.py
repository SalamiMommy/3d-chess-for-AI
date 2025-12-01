
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.common.shared_types import PieceType, Color, COORD_DTYPE, PIECE_TYPE_DTYPE, COLOR_DTYPE
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.cache.manager import OptimizedCacheManager
from game3d.movement.pseudolegal import generate_pseudolegal_moves_batch

def test_spiral_occupancy():
    print("Testing Spiral Occupancy...")
    
    # 1. Setup Board
    board = Board.empty()
    empty_coords = np.empty((0, 3), dtype=COORD_DTYPE)
    empty_types = np.empty(0, dtype=PIECE_TYPE_DTYPE)
    empty_colors = np.empty(0, dtype=COLOR_DTYPE)
    
    cache = OptimizedCacheManager(board, Color.WHITE, initial_data=(empty_coords, empty_types, empty_colors))
    state = GameState(board=board, color=Color.WHITE, cache_manager=cache)
    
    # Place White Spiral at center (4, 4, 4)
    center = np.array([4, 4, 4], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(center, (PieceType.SPIRAL, Color.WHITE))
    
    # Place Black Pawn blocking one ray (e.g. +X direction)
    # Spiral +X ray: [1, 0, 0]
    blocker_pos = np.array([6, 4, 4], dtype=COORD_DTYPE) # 2 steps away
    cache.occupancy_cache.set_position(blocker_pos, (PieceType.PAWN, Color.BLACK))
    
    # Place White Pawn blocking another ray (e.g. +Y direction)
    # Spiral +Y ray: [0, 1, 0]
    friendly_pos = np.array([4, 6, 4], dtype=COORD_DTYPE) # 2 steps away
    cache.occupancy_cache.set_position(friendly_pos, (PieceType.PAWN, Color.WHITE))
    
    # Ensure generator is initialized (if needed, though we call pseudolegal directly)
    from game3d.movement import generator
    generator.initialize_generator()
    
    # 2. Generate Moves with ignore_occupancy=False (Default/Pseudolegal)
    print("\n--- Generating Pseudolegal Moves (ignore_occupancy=False) ---")
    moves_blocked = generate_pseudolegal_moves_batch(
        state, 
        np.array([center]), 
        ignore_occupancy=False
    )
    
    # Check if blocked by Black Pawn (should capture but stop)
    # Ray: (5,4,4), (6,4,4) [Capture], (7,4,4) [Blocked]
    # So (7,4,4) should NOT be in moves. (6,4,4) SHOULD be.
    
    has_capture = np.any(np.all(moves_blocked[:, 3:] == blocker_pos, axis=1))
    has_past_blocker = np.any(np.all(moves_blocked[:, 3:] == np.array([7, 4, 4]), axis=1))
    
    print(f"Blocked by Enemy at {blocker_pos}: Capture={has_capture}, Past={has_past_blocker}")
    
    if has_past_blocker:
        print("FAIL: Pseudolegal moves went through enemy piece!")
    else:
        print("PASS: Pseudolegal moves stopped at enemy piece.")
        
    # Check if blocked by White Pawn (should stop BEFORE)
    # Ray: (4,5,4), (4,6,4) [Blocked]
    # So (4,6,4) should NOT be in moves. (4,5,4) SHOULD be.
    
    has_friendly = np.any(np.all(moves_blocked[:, 3:] == friendly_pos, axis=1))
    has_past_friendly = np.any(np.all(moves_blocked[:, 3:] == np.array([4, 7, 4]), axis=1))
    
    print(f"Blocked by Friendly at {friendly_pos}: Capture={has_friendly}, Past={has_past_friendly}")
    
    if has_friendly or has_past_friendly:
        print("FAIL: Pseudolegal moves went through/captured friendly piece!")
    else:
        print("PASS: Pseudolegal moves stopped before friendly piece.")


    # 3. Generate Moves with ignore_occupancy=True (Raw)
    print("\n--- Generating Raw Moves (ignore_occupancy=True) ---")
    moves_raw = generate_pseudolegal_moves_batch(
        state, 
        np.array([center]), 
        ignore_occupancy=True
    )
    
    # Check if goes through Black Pawn
    has_past_blocker_raw = np.any(np.all(moves_raw[:, 3:] == np.array([7, 4, 4]), axis=1))
    
    print(f"Blocked by Enemy at {blocker_pos} (Raw): Past={has_past_blocker_raw}")
    
    if not has_past_blocker_raw:
        print("FAIL: Raw moves did NOT go through enemy piece!")
    else:
        print("PASS: Raw moves went through enemy piece.")
        
    # Check if goes through White Pawn
    has_past_friendly_raw = np.any(np.all(moves_raw[:, 3:] == np.array([4, 7, 4]), axis=1))
    
    print(f"Blocked by Friendly at {friendly_pos} (Raw): Past={has_past_friendly_raw}")
    
    if not has_past_friendly_raw:
        print("FAIL: Raw moves did NOT go through friendly piece!")
    else:
        print("PASS: Raw moves went through friendly piece.")

if __name__ == "__main__":
    test_spiral_occupancy()
