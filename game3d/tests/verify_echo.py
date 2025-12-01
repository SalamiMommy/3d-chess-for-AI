import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.pieces.pieces.echo import generate_echo_moves, _ECHO_DIRECTIONS
from game3d.common.shared_types import RADIUS_1_OFFSETS
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType

def verify_echo_modification():
    print("Verifying Echo modification...")
    
    # Check if _ECHO_DIRECTIONS has the expected shape
    # 6 anchors * 26 bubbles = 156
    expected_count = 6 * 26
    actual_count = _ECHO_DIRECTIONS.shape[0]
    
    print(f"Expected direction count: {expected_count}")
    print(f"Actual direction count: {actual_count}")
    
    if expected_count != actual_count:
        print("FAIL: Direction count mismatch")
        return False
        
    # Verify anchors are at distance 2 (cardinal)
    anchors = np.array([
        [-2, 0, 0], [2, 0, 0],
        [0, -2, 0], [0, 2, 0],
        [0, 0, -2], [0, 0, 2]
    ])
    # Verify bubbles are at distance 1
    bubbles = RADIUS_1_OFFSETS
    
    # Reconstruct expected directions manually
    expected_directions = []
    for anchor in anchors:
        for bubble in bubbles:
            expected_directions.append(anchor + bubble)
    
    expected_directions = np.array(expected_directions)
    
    # Check if all expected directions are present in _ECHO_DIRECTIONS
    # Note: order might differ, so we sort or use set logic if needed, 
    # but since the implementation uses the same logic, they should match exactly if reshaped.
    
    # Let's just check if the unique vectors match
    unique_expected = np.unique(expected_directions, axis=0)
    unique_actual = np.unique(_ECHO_DIRECTIONS, axis=0)
    
    print(f"Unique expected vectors: {len(unique_expected)}")
    print(f"Unique actual vectors: {len(unique_actual)}")
    
    if len(unique_expected) != len(unique_actual):
        print("FAIL: Unique vector count mismatch")
        return False
        
    if not np.array_equal(unique_expected, unique_actual):
        print("FAIL: Unique vectors do not match")
        return False

    print("PASS: Echo directions verified")
    return True

if __name__ == "__main__":
    if verify_echo_modification():
        sys.exit(0)
    else:
        sys.exit(1)
