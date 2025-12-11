import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType
from game3d.attacks.check import king_in_check, get_attackers
from game3d.common.coord_utils import coords_to_keys

def test_ghost_attack_desync():
    print("Setting up game state...")
    board = Board.empty()
    gs = GameState(board, Color.WHITE)
    
    # 0. Ensure OccupancyCache is EMPTY using rebuild
    gs.cache_manager.occupancy_cache.rebuild(
        np.empty((0, 3), dtype=np.int16),
        np.empty(0, dtype=np.int8),
        np.empty(0, dtype=np.int8)
    )
    # Clear move cache too
    gs.cache_manager.move_cache.invalidate()

    # Also verify it is empty
    assert len(gs.cache_manager.occupancy_cache.get_positions(Color.WHITE)) == 0
    assert len(gs.cache_manager.occupancy_cache.get_positions(Color.BLACK)) == 0
    
    # 1. Setup: White King at (0,0,0), Black Rook at (0,0,5)
    # This puts White in check.
    king_pos = np.array([0, 0, 0], dtype=np.int16)
    rook_pos = np.array([0, 0, 5], dtype=np.int16)
    
    gs.cache_manager.occupancy_cache.set_position(
        king_pos, 
        np.array([PieceType.KING.value, Color.WHITE.value], dtype=np.int8)
    )
    gs.cache_manager.occupancy_cache.set_position(
        rook_pos, 
        np.array([PieceType.ROOK.value, Color.BLACK.value], dtype=np.int8)
    )
    
    # ensure cache is fresh
    gs.cache_manager.move_cache.invalidate()
    
    # 2. Trigger Move Generation & Cache Population
    # Calling king_in_check should populate the cache (Bitboard + Moves)
    is_check = king_in_check(gs.board, Color.WHITE, Color.WHITE, gs.cache_manager)
    print(f"Initial Check Status: {is_check}")
    
    # Debug: Check occupancy
    k_ptype, k_color = gs.cache_manager.occupancy_cache.get(king_pos).values()
    r_ptype, r_color = gs.cache_manager.occupancy_cache.get(rook_pos).values()
    print(f"King at {king_pos}: Type={k_ptype}, Color={k_color}")
    print(f"Rook at {rook_pos}: Type={r_ptype}, Color={r_color}")
    
    # Debug: Check if move generation finds the rook move
    moves = gs.cache_manager.move_cache.get_pseudolegal_moves(Color.BLACK)
    if moves is None:
        print("Moves for BLACK not cached yet inside king_in_check call?")
        # Force generate
        from game3d.core.buffer import state_to_buffer
        from game3d.core.api import generate_pseudolegal_moves
        from game3d.common.shared_types import MinimalStateProxy
        dummy_state = MinimalStateProxy(gs.board, Color.BLACK.value, gs.cache_manager)
        buffer = state_to_buffer(dummy_state, readonly=True)
        moves = generate_pseudolegal_moves(buffer)
        print(f"Generated {len(moves)} moves for BLACK.")
        
    found_attack = False
    for m in moves:
         if m[3]==0 and m[4]==0 and m[5]==0:
             print(f"Found attack: {m}")
             found_attack = True
    print(f"Found attack in moves: {found_attack}")

    assert is_check == True, "Should be in check initially"
    
    attackers = get_attackers(gs)
    print(f"Initial Attackers: {attackers}")
    assert len(attackers) > 0, "Should find attackers initially"
    
    # 3. Simulate invalidation of the Rook (as if it moved or was captured)
    # We mark it as invalid in MoveCache.
    # This simulates the "Incremental Update" phase where we've marked dirty pieces
    # but haven't fully regenerated the global state yet.
    
    # Get Rook key
    rook_keys = coords_to_keys(np.array([rook_pos]))
    gs.cache_manager.move_cache.mark_piece_invalid(Color.BLACK, rook_keys[0])
    
    print("\nMarked Rook as invalid (dirty).")
    
    # 4. Check for Desync
    # MoveCache now has a dirty piece.
    # get_pseudolegal_moves should return None (miss).
    moves = gs.cache_manager.move_cache.get_pseudolegal_moves(Color.BLACK)
    print(f"Pseudolegal Moves Cache Hit: {moves is not None}")
    
    if moves is None:
        print("Correct: Pseudolegal moves cache invalidated by dirty piece.")
    else:
        print("FAIL: Pseudolegal moves cache still valid despite dirty piece.")
        
    # BUT, is_under_attack (Bitboard) uses the STALE bitboard?
    # king_in_check calls square_attacked_by -> move_cache.is_under_attack
    
    print("\nChecking king safety with dirty cache...")
    try:
        # We manually check the bitboard status via is_under_attack
        # The bitboard should theoretically be considered 'invalid' or 'dirty' if there are affected pieces.
        stale_check = gs.cache_manager.move_cache.is_under_attack(king_pos, Color.WHITE)
        print(f"MoveCache.is_under_attack((0,0,0)): {stale_check}")
        
        if stale_check:
            print("FAIL: Bitboard reported Attack despite dirty source piece! (Ghost Attack)")
        else:
            print("SUCCESS: Bitboard correctly reported False/Safe due to dirty state.")
            
    except Exception as e:
        print(f"Error during check: {e}")

    # 5. Verify consistency with get_attackers
    # get_attackers forces regeneration if cache returns None.
    # So if we moved the rook (removed it for this test), get_attackers should see NO attackers.
    
    # Let's actually remove the rook from board to make the "Ghost" real
    gs.cache_manager.occupancy_cache.set_position(rook_pos, None)
    
    # Note: We already marked it invalid. Now we really removed it.
    # If is_under_attack relies on the OLD bitboard, it will still say True.
    # But get_attackers (regenerating from current board) will say Empty.
    
    print("\nRemoved Rook from board (Ghost Scenario).")
    ghost_check = king_in_check(gs.board, Color.WHITE, Color.WHITE, gs.cache_manager)
    print(f"is_check (King In Check): {ghost_check}")
    
    ghost_attackers = get_attackers(gs)
    print(f"get_attackers: {ghost_attackers}")
    
    if ghost_check and not ghost_attackers:
        print("\n!!! REPRODUCTION SUCCESSFUL: Ghost Attack Detected !!!")
        print("is_check=True (Stale Cache) vs get_attackers=[] (Real State)")
    else:
        print("\nReproduction Failed (Behavior is consistent).")

if __name__ == "__main__":
    test_ghost_attack_desync()
