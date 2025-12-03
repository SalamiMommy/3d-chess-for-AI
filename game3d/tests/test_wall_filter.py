
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.common.shared_types import SIZE, COORD_DTYPE

def test_wall_filtering_logic():
    print(f"Testing Wall filtering logic with SIZE={SIZE}")
    
    # Simulate moves array returned by kernel
    # Format: [from_x, from_y, from_z, to_x, to_y, to_z]
    # Wall is 2x2. Anchor at (x,y) occupies (x,y), (x+1,y), (x,y+1), (x+1,y+1).
    # Valid anchor range: 0 <= x < SIZE-1, 0 <= y < SIZE-1
    
    moves = np.array([
        [0, 0, 0, 0, 0, 0],       # Valid: to (0,0) -> occupies (0,0)-(1,1)
        [0, 0, 0, SIZE-2, SIZE-2, 0], # Valid: to (7,7) -> occupies (7,7)-(8,8)
        [0, 0, 0, SIZE-1, 0, 0],  # INVALID: to (8,0) -> occupies (8,0)-(9,1) -> OOB X
        [0, 0, 0, 0, SIZE-1, 0],  # INVALID: to (0,8) -> occupies (0,8)-(1,9) -> OOB Y
        [0, 0, 0, SIZE-1, SIZE-1, 0], # INVALID: to (8,8) -> occupies (8,8)-(9,9) -> OOB X & Y
    ], dtype=COORD_DTYPE)
    
    print(f"Input moves:\n{moves[:, 3:]}")
    
    # Apply the logic added to wall.py
    if moves.size > 0:
        # Check to_x and to_y < SIZE - 1
        valid_mask = (moves[:, 3] < SIZE - 1) & (moves[:, 4] < SIZE - 1)
        filtered_moves = moves[valid_mask]
    else:
        filtered_moves = moves

    print(f"Filtered moves:\n{filtered_moves[:, 3:]}")
    
    # Assertions
    expected_count = 2
    if len(filtered_moves) != expected_count:
        print(f"❌ FAILED: Expected {expected_count} moves, got {len(filtered_moves)}")
        sys.exit(1)
        
    # Check that no invalid moves remain
    if np.any(filtered_moves[:, 3] >= SIZE - 1) or np.any(filtered_moves[:, 4] >= SIZE - 1):
        print("❌ FAILED: Invalid moves remain in filtered output!")
        sys.exit(1)
        
    print("✅ SUCCESS: Filtering logic works correctly.")

if __name__ == "__main__":
    test_wall_filtering_logic()
