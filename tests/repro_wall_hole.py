
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE
from game3d.game.turnmove import _process_hole_effects

def test_wall_blackhole_prohibition():
    """Test that a Blackhole does NOT suck a Wall."""
    print("Initializing GameState...")
    from game3d.board.board import Board
    state = GameState(Board())
    cache_manager = state.cache_manager
    
    # 1. Setup: Clear board
    # We can't clear easily without clearing cache, so let's just use empty spots
    # Assuming start position, 4,7,6 and 4,8,6 might be empty or not.
    
    # Place White Wall at [4, 7, 0] (Using Z=0 to be safe from other pieces)
    # Wall is 2x2. Anchor at [4, 7, 0]. Occupies (4,7,0), (5,7,0), (4,8,0), (5,8,0).
    # Wait, (4,8,0) means Y=8.
    # SIZE=9. Max index=8.
    # So Y=8 is valid for PART of the wall, but NOT for the anchor if it extends +1?
    # Wall anchor at [4, 7] -> (4,7), (5,7), (4,8), (5,8).
    # All are <= 8. Correct.
    
    wall_anchor = np.array([4, 7, 0], dtype=COORD_DTYPE)
    # Set Wall pieces manually
    from game3d.pieces.pieces.wall import WALL_BLOCK_OFFSETS
    
    print(f"Placing White Wall at anchor {wall_anchor} ...")
    for off in WALL_BLOCK_OFFSETS:
        pos = wall_anchor + off
        cache_manager.occupancy_cache.set_position(pos, [PieceType.WALL, Color.WHITE])
        
    # Place Black Blackhole at [4, 8, 2]
    # Distance to [4, 8, 0] is 2 (Z diff).
    # Distance to [4, 7, 0] is max(0, 1, 2) = 2.
    # Blackhole Pull Radius is 2.
    # So it is within range.
    
    hole_pos = np.array([4, 8, 2], dtype=COORD_DTYPE)
    print(f"Placing Black Blackhole at {hole_pos} ...")
    cache_manager.occupancy_cache.set_position(hole_pos, [PieceType.BLACKHOLE, Color.BLACK])
    
    # Verify setup
    w_type = cache_manager.occupancy_cache.get_type_at(*wall_anchor)
    print(f"Type at wall anchor {wall_anchor}: {w_type} (Expected {PieceType.WALL})")
    
    # 2. Run Hole Effects
    # Moving player is BLACK (who owns the hole)
    print("Running _process_hole_effects(state, BLACK)...")
    _process_hole_effects(state, Color.BLACK)
    
    # 3. Check if Wall moved
    # If sucked, it would move towards hole.
    # Direction from Wall [4,7,0] to Hole [4,8,2]:
    # dx=0, dy=1, dz=2.
    # Step: (0, 1, 1).
    # New Pos: [4, 8, 1].
    
    new_type_at_anchor = cache_manager.occupancy_cache.get_type_at(*wall_anchor)
    type_at_target = cache_manager.occupancy_cache.get_type_at(4, 8, 1)
    
    print(f"Type at original anchor {wall_anchor}: {new_type_at_anchor}")
    print(f"Type at pull target [4, 8, 1]: {type_at_target}")
    
    if new_type_at_anchor == PieceType.WALL:
        print("SUCCESS: Wall did not move.")
    else:
        print("FAILURE: Wall moved! (or was removed)")
        
    # Also check if it moved to target
    if type_at_target == PieceType.WALL:
        print("FAILURE: Wall found at pull target!")

if __name__ == "__main__":
    test_wall_blackhole_prohibition()
