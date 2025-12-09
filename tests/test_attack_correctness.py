
import time
import numpy as np
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import PieceType, SIZE
from game3d.attacks.attack_registry import square_attacked_by_extended

def test_xyqueen_attack_patterns():
    """Verify specific attack patterns for XYQUEEN (Slider in XY, King in 3D)."""
    gs = GameState(Board.startpos())
    gs.cache_manager.occupancy_cache.clear()
    
    # Place White XYQUEEN at (0,0,0)
    coord = np.array([0, 0, 0], dtype=np.int16)
    piece_data = np.array([PieceType.XYQUEEN.value, 1], dtype=np.int8) 
    gs.cache_manager.occupancy_cache.set_position(coord, piece_data)
    gs.cache_manager.move_cache.invalidate()
    
    attacker_color = 1
    
    # Test Cases
    cases = [
        # (Target, Duration?, ShouldBeAttacked, Description)
        ([0, 8, 0], True, "Long range XY Orthogonal"),
        ([8, 8, 0], True, "Long range XY Diagonal"),
        ([0, 1, 0], True, "Short range XY Orthogonal (covered by Slider & King)"),
        ([0, 0, 1], True, "Short range Z Orthogonal (King move)"),
        ([1, 0, 1], True, "Short range XZ Diagonal (King move)"),
        ([0, 0, 8], False, "Long range Z Orthogonal (Should NOT attack)"),
        ([2, 0, 2], False, "Long range XZ Diagonal (Should NOT attack)"),
        ([0, 0, 2], False, "Medium range Z Orthogonal (Should NOT attack)"),
    ]
    
    for target_list, expected, desc in cases:
        target = np.array(target_list, dtype=np.int16)
        
        # Ensure target is valid coord
        if np.any(target < 0) or np.any(target >= SIZE):
            print(f"Skipping invalid target {target}")
            continue
            
        is_attacked = square_attacked_by_extended(
            gs.board, target, attacker_color, gs.cache_manager
        )
        
        status = "PASS" if is_attacked == expected else "FAIL"
        print(f"[{status}] {desc} {target}: Got {is_attacked}, Expected {expected}")
        
        if is_attacked != expected:
            raise AssertionError(f"Failed {desc}: {target}")

if __name__ == "__main__":
    try:
        test_xyqueen_attack_patterns()
        print("\nAll XYQUEEN correctness tests passed!")
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        exit(1)
