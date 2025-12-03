
import pytest
import numpy as np
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color, SIZE
from game3d.pieces.pieces.wall import generate_wall_moves

def test_wall_boundary_moves():
    """Test Wall moves near all boundaries to ensure no OOB moves are generated."""
    game = GameState.from_startpos()
    game.cache_manager.occupancy_cache.clear()
    
    # Test cases: (anchor_pos, description)
    # Wall is 2x2, so anchor at (x,y) occupies (x,y), (x+1,y), (x,y+1), (x+1,y+1)
    # Max valid anchor is (SIZE-2, SIZE-2, SIZE-1)
    # For SIZE=9, max anchor is (7, 7, 8)
    
    test_anchors = [
        (7, 7, 5), # Corner in XY plane
        (7, 0, 5), # Edge X
        (0, 7, 5), # Edge Y
        (5, 5, 8), # Top Z
        (5, 5, 0), # Bottom Z
        (6, 6, 6), # Near corner
    ]
    
    for x, y, z in test_anchors:
        anchor = np.array([x, y, z], dtype=np.int16)
        
        # Set up the wall
        # We need to ensure it's a valid anchor (left and up neighbors empty)
        # Since board is clear, this is guaranteed.
        
        # Manually place wall parts to be realistic (though generator only checks anchor validity via neighbors)
        parts = [
            (x, y, z), (x+1, y, z),
            (x, y+1, z), (x+1, y+1, z)
        ]
        
        for px, py, pz in parts:
            game.cache_manager.occupancy_cache.set_position(
                np.array([px, py, pz], dtype=np.int16), 
                np.array([PieceType.WALL, Color.WHITE])
            )
            
        moves = generate_wall_moves(game.cache_manager, Color.WHITE, anchor)
        
        # Verify all moves are valid
        for i in range(len(moves)):
            m = moves[i]
            tx, ty, tz = m[3], m[4], m[5]
            
            # Check anchor bounds
            assert 0 <= tx < SIZE - 1, f"Invalid anchor X: {tx} for move from {anchor}"
            assert 0 <= ty < SIZE - 1, f"Invalid anchor Y: {ty} for move from {anchor}"
            assert 0 <= tz < SIZE, f"Invalid anchor Z: {tz} for move from {anchor}"
            
            # Check parts bounds (implicit in anchor check, but good to be explicit)
            assert tx + 1 < SIZE
            assert ty + 1 < SIZE
            
        # Clear for next test
        game.cache_manager.occupancy_cache.clear()

if __name__ == "__main__":
    test_wall_boundary_moves()
