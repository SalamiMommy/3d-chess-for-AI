import numpy as np
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType
from game3d.cache.manager import OptimizedCacheManager as CacheManager
from game3d.attacks.check import square_attacked_by

def test_ghost_attack_move_repro():
    print("Initializing Board and Cache...")
    board = Board()
    cache = CacheManager(board)
    board.cache_manager = cache
    
    # 1. Setup scenario:
    # White King at (2, 6, 2) - Attacking (3, 6, 2)
    # Black King at (3, 6, 2)
    # Distance is 1, so Black King IS attacked.
    
    cache.occupancy_cache.clear()
    
    wk_start = np.array([2, 6, 2])
    bk_pos = np.array([3, 6, 2])
    
    cache.occupancy_cache.set_position(wk_start, np.array([PieceType.KING, Color.WHITE]))
    cache.occupancy_cache.set_position(bk_pos, np.array([PieceType.KING, Color.BLACK]))
    
    print(f"Setup: WK at {wk_start}, BK at {bk_pos}")
    
    # 2. Populate Cache (Full Generation)
    print("Populating initial cache...")
    from game3d.core.api import generate_pseudolegal_moves
    from game3d.core.buffer import state_to_buffer
    from game3d.common.shared_types import MinimalStateProxy
    
    # Generate White moves to populate White's attack bitboard
    proxy = MinimalStateProxy(board, Color.WHITE, cache)
    moves = generate_pseudolegal_moves(state_to_buffer(proxy))
    cache.move_cache.store_pseudolegal_moves(Color.WHITE, moves)
    
    # Verify Initial Check
    is_attacked = square_attacked_by(board, Color.BLACK, bk_pos, Color.WHITE.value, cache)
    print(f"Initial Check: Black King attacked? {is_attacked} (Expected: True)")
    if not is_attacked:
        print("FAIL: Setup incorrect, BK should be attacked.")
        return

    # 3. Move White King to (1, 6, 3)
    # New distance to (3, 6, 2):
    # |1-3|=2, |6-6|=0, |3-2|=1. Max dist = 2.
    # So BK should NOT be attacked anymore.
    
    wk_end = np.array([1, 6, 3])
    print(f"Moving WK to {wk_end}...")
    
    # Update Occupancy
    cache.occupancy_cache.set_position(wk_start, None)
    cache.occupancy_cache.set_position(wk_end, np.array([PieceType.KING, Color.WHITE]))
    
    # mark_piece_invalid (simulating Incremental Update)
    # We must mark the old key invalid
    wk_key = wk_start[0] | (wk_start[1] << 9) | (wk_start[2] << 18)
    cache.move_cache.mark_piece_invalid(Color.WHITE, wk_key)
    
    # 4. Regenerate Moves for White King (Incremental)
    # In real game, get_incremental_state would identify it needs regen
    # Here we simulate the regen and store
    
    # Simulate calculating new moves for WK at wk_end
    # (Simplified: assume we know the moves, or use generator)
    # We'll just generate moves for the whole board again but only store for the piece
    # This is inefficient but functional for repro
    proxy_new = MinimalStateProxy(board, Color.WHITE, cache)
    moves_new = generate_pseudolegal_moves(state_to_buffer(proxy_new))
    
    # Filter for King moves only
    wk_moves = moves_new[
        (moves_new[:, 0] == wk_end[0]) & 
        (moves_new[:, 1] == wk_end[1]) & 
        (moves_new[:, 2] == wk_end[2])
    ]
    
    # New Piece Key
    wk_new_key = wk_end[0] | (wk_end[1] << 9) | (wk_end[2] << 18)
    
    print(f"Storing new moves for WK (count: {len(wk_moves)})...")
    cache.move_cache.store_piece_moves(Color.WHITE, wk_new_key, wk_moves)
    
    # 5. Clear affected pieces (Simulating end of incremental update step)
    cache.move_cache.clear_affected_pieces(Color.WHITE)
    
    # 6. Check for Ghost Attack
    print("\nChecking if Black King is still attacked (Attackers should be NONE)...")
    is_attacked_after = square_attacked_by(board, Color.BLACK, bk_pos, Color.WHITE.value, cache)
    print(f"Black King attacked? {is_attacked_after}")
    
    if is_attacked_after:
        print("\n[SUCCESS REPRO] GHOST ATTACK DETECTED!")
        print("The White King moved away, but the cache still reports an attack.")
    else:
        print("\n[FAIL REPRO] No ghost attack. Cache correctly updated.")

if __name__ == "__main__":
    test_ghost_attack_move_repro()
