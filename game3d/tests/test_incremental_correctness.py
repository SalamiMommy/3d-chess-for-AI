
import sys
import os
import numpy as np

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game3d.common.shared_types import Color, COORD_DTYPE, PieceType
from game3d.game.factory import start_game_state
from game3d.main_game import OptimizedGame3D
from game3d.attacks.check import square_attacked_by_incremental

def test_incremental_correctness():
    print("Testing square_attacked_by_incremental Correctness...")
    
    # 1. Setup Game
    gs = start_game_state()
    # Clear board for controlled test
    # (assuming we can manipulate cache/board directly)
    # Actually, let's just use empty board or clear it.
    
    cache = gs.cache_manager
    occ = cache.occupancy_cache
    
    # Clear all pieces
    # We can use _initialize_from_setup with empty lists if we want, or just verify a specific interaction
    # Let's try to find an empty spot or just clear manually (might be dirty)
    
    # Better: Create a scenario on the existing board.
    # Find a square that IS currently attacked by exactly one piece.
    # Move that piece away.
    # Verify it's no longer attacked.
    
    # Let's place a CUSTOM piece setup.
    # We can use `rebuild` on occupancy cache.
    white_rook_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    white_rook_type = np.array([PieceType.ROOK], dtype=np.int8) # Assuming ROOK is valid int
    white_indices = np.array([Color.WHITE], dtype=np.int8)
    
    coords = np.array([white_rook_pos], dtype=COORD_DTYPE)
    types = np.array([PieceType.ROOK], dtype=np.int8)
    colors = np.array([Color.WHITE], dtype=np.int8)
    
    occ.rebuild(coords, types, colors)
    # Also need to clear/reset move cache
    cache.move_cache.invalidate()
    
    # Ensure cache is populated for the "Before" state
    # We must explicitly generate moves to populate the cache (MoveCache is passive)
    from game3d.movement.generator import generate_legal_moves
    generate_legal_moves(gs)
    
    # Target square: (0, 0, 5) should be attacked by Rook at (0,0,0)
    target = np.array([0, 0, 5], dtype=COORD_DTYPE)
    
    # Verify initial state (using full check)
    from game3d.attacks.check import square_attacked_by
    initial_attacked = square_attacked_by(gs.board, Color.BLACK, target, Color.WHITE.value, cache)
    print(f"Initial: Square {target} attacked by White? {initial_attacked}")
    
    # DEBUG: Check MoveCache state
    print("Debug: Checking MoveCache internal state...")
    try:
        attack_mask = cache.move_cache.get_attack_mask(Color.WHITE.value)
        if attack_mask is None:
             print("Debug: get_attack_mask returned None (Dirty/Not Rebuilt). Skipping check.")
        elif not attack_mask[0, 0, 5]:
             print("Debug: Attack mask says (0,0,5) is NOT attacked! (Cache/Matrix issue)")
    except Exception as e:
        print(f"Debug: get_attack_mask failed: {e}")
    
    # DEBUG: Check affected pieces for our simulated moves
    affected, _, _ = cache.move_cache.get_pieces_affected_by_move(
         np.array([0,0,0], dtype=COORD_DTYPE),
         np.array([1,0,0], dtype=COORD_DTYPE),
         Color.WHITE.value
    )
    print(f"Debug: Affected pieces for move (0,0,0)->(1,0,0): {len(affected)}")
    
    affected2, _, _ = cache.move_cache.get_pieces_affected_by_move(
         np.array([0,0,0], dtype=COORD_DTYPE),
         np.array([0,0,2], dtype=COORD_DTYPE),
         Color.WHITE.value
    )
    print(f"Debug: Affected pieces for move (0,0,0)->(0,0,2): {len(affected2)}")

    if not initial_attacked:
        print("FAILED: Setup invalid, target should be attacked initially.")
        return False

    # 2. Simulate Move: Rook (0,0,0) -> (1,0,0)
    # This removes attack on (0,0,5) (assuming standard rook moves orthogonal)
    # Rook at (1,0,0) attacks (1,0,z), (1,y,0), (x,0,0)...
    # Does it attack (0,0,5)? No.
    
    from_coord = np.array([0, 0, 0], dtype=COORD_DTYPE)
    to_coord = np.array([1, 0, 0], dtype=COORD_DTYPE)
    
    # 3. Call Incremental Check
    # "is target attacked by WHITE after this move?"
    is_attacked_after = square_attacked_by_incremental(
        gs.board,
        target,
        Color.WHITE.value, # Attacker
        cache,
        from_coord,
        to_coord
    )
    
    print(f"After Move (0,0,0)->(1,0,0): Target {target} attacked? {is_attacked_after}")
    
    if is_attacked_after:
        print("FAILED: Target still reported as attacked after attacker moved away.")
        return False
        
    # 4. Simulate Move: Rook (0,0,0) -> (0,0,2)
    # This KEEPS attack on (0,0,5) (still on z-axis)
    to_coord_2 = np.array([0, 0, 2], dtype=COORD_DTYPE)
    
    is_attacked_after_2 = square_attacked_by_incremental(
        gs.board,
        target,
        Color.WHITE.value,
        cache,
        from_coord,
        to_coord_2
    )
    
    print(f"After Move (0,0,0)->(0,0,2): Target {target} attacked? {is_attacked_after_2}")
    
    if not is_attacked_after_2:
        print("FAILED: Target reported as SAFE (not attacked) but attacker moved along ray (should still attack).")
        return False
        
    print("SUCCESS: Incremental check logic verified.")
    return True

if __name__ == "__main__":
    success = test_incremental_correctness()
    sys.exit(0 if success else 1)
