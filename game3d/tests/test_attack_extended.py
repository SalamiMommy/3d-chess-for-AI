
import sys
import os
import numpy as np
import pytest

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game3d.game.factory import start_game_state
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE
from game3d.attacks.attack_registry import square_attacked_by_extended

def test_attack_basic():
    """Verify basic attack detection (Queens and Pawns)."""
    gs = start_game_state(ensure_start_pos=False)
    cache = gs.cache_manager
    
    # Place a White Queen at 4,4,4
    queen_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(queen_pos, (PieceType.QUEEN, Color.WHITE))
    
    # Check attack on adjacent square (5,4,4)
    target = np.array([5, 4, 4], dtype=COORD_DTYPE)
    assert square_attacked_by_extended(gs.board, target, Color.WHITE, cache)
    
    # Check attack on diagonal (5,5,5)
    target = np.array([5, 5, 5], dtype=COORD_DTYPE)
    assert square_attacked_by_extended(gs.board, target, Color.WHITE, cache)
    
    # Check NO attack on safe square (4,4,4 itself is occupied, but let's check non-attacked)
    # The Queen at 4,4,4 should NOT attack 0,0,0 (unless 9x9 board allows it... wait, 4,4,4 to 0,0,0 is 4,4,4 vector)
    # 4,4,4 -> 0,0,0: dx=-4, dy=-4, dz=-4. Yes, it IS attacked by Queen!
    
    # Let's pick a knight-jump away square which Queen CANNOT hit: 4+1, 4+2, 4 = 5,6,4
    target = np.array([5, 6, 4], dtype=COORD_DTYPE)
    assert not square_attacked_by_extended(gs.board, target, Color.WHITE, cache)

def test_attack_pawn():
    """Verify Pawn attack logic."""
    gs = start_game_state(ensure_start_pos=False)
    cache = gs.cache_manager
    
    # White Pawn at 4,4,4
    # Attacks 3,5,5 and 5,5,5? No, White Pawns move +Z (or whatever direction).
    # Standard white pawn attacks: z+1, and x/y adjacent?
    # Let's check kernel logic:
    # if attacker_color == 1 (White): if dz == 1 and abs(tx - ax) == 1 and abs(ty - ay) == 1:
    # So it attacks diagonals in Z+1 plane.
    
    pawn_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(pawn_pos, (PieceType.PAWN, Color.WHITE))
    
    # Valid attack: 5,5,5 (dx=1, dy=1, dz=1)
    target = np.array([5, 5, 5], dtype=COORD_DTYPE)
    assert square_attacked_by_extended(gs.board, target, Color.WHITE, cache)
    
    # Invalid attack: 4,4,5 (directly above - move but not attack)
    target = np.array([4, 4, 5], dtype=COORD_DTYPE)
    assert not square_attacked_by_extended(gs.board, target, Color.WHITE, cache)
    
    # Invalid attack: 5,5,3 (behind)
    target = np.array([5, 5, 3], dtype=COORD_DTYPE)
    assert not square_attacked_by_extended(gs.board, target, Color.WHITE, cache)

def test_attack_fallback():
    """Verify fallback mechanism for Jumpers (Knight)."""
    # Assuming Knight is handled by generic jump table or fallback
    gs = start_game_state(ensure_start_pos=False)
    cache = gs.cache_manager
    
    knight_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    cache.occupancy_cache.set_position(knight_pos, (PieceType.KNIGHT, Color.WHITE))
    
    # Valid Jump: +1, +2, 0 -> 5, 6, 4
    target = np.array([5, 6, 4], dtype=COORD_DTYPE)
    assert square_attacked_by_extended(gs.board, target, Color.WHITE, cache)
    
    # Invalid Jump
    target = np.array([5, 5, 5], dtype=COORD_DTYPE)
    assert not square_attacked_by_extended(gs.board, target, Color.WHITE, cache)

if __name__ == "__main__":
    try:
        test_attack_basic()
        print("Basic Attack: PASS")
        test_attack_pawn()
        print("Pawn Attack: PASS")
        test_attack_fallback()
        print("Fallback/Jump Attack: PASS")
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
