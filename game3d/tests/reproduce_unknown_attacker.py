import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import Color, PieceType
from game3d.game.terminal import is_check
from game3d.attacks.check import get_attackers

logging.basicConfig(level=logging.INFO)

def reproduce():
    print("Initializing Board...")
    
    # Create scratch board
    board = Board.startpos()
    
    # Create game state
    game = GameState(board, Color.WHITE)
    
    occ_cache = game.cache_manager.occupancy_cache
    
    # Clear board
    print("Clearing board...")
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    
    # Need to collect all coords first to avoid modification during iteration issues if any
    # But get_all_occupied returns a copy usually.
    coords_list = [coords[i] for i in range(len(coords))]
    
    for c in coords_list:
        occ_cache.set_position(c, None)
        
    # Verify empty
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    assert len(coords) == 0, "Board not empty"
    
    # Setup Checkmate scenario
    print("Setting up check scenario...")
    # Add White King at 2,2,2
    w_king_pos = np.array([2, 2, 2], dtype=np.int32)
    occ_cache.set_position(w_king_pos, np.array([PieceType.KING, Color.WHITE]))
    
    # Add Black Rook at 2,2,5 (attacking)
    b_rook_pos = np.array([2, 2, 5], dtype=np.int32)
    occ_cache.set_position(b_rook_pos, np.array([PieceType.ROOK, Color.BLACK]))
    
    # Force cache invalidation explicitly to be safe
    # (Though occ_cache should notify manager -> move_cache)
    game.cache_manager.move_cache.invalidate_pseudolegal_moves(None)
    
    # Corrupt the cache: Set cached moves to empty but valid
    print("Corrupting cache to simulate desync...")
    empty_moves = np.empty((0, 6), dtype=np.int16)
    
    # Manually inject empty moves into cache and lie about validity
    # We need to set internal state of move_cache
    mc = game.cache_manager.move_cache
    b_color_idx = 1 # Black
    
    # Set pseudolegal moves to empty
    mc._pseudolegal_moves_cache[b_color_idx] = empty_moves
    
    # Clear affected lists to make it look "fresh"
    mc._affected_keys_per_color[b_color_idx] = []
    
    # Invalidate (clear) bitboards
    mc._attack_bitboards[b_color_idx] = None 
    mc._bitboard_dirty[b_color_idx] = True 
    
    print("Getting attackers (should find via fallback despite corrupted cache)...")
    attackers = get_attackers(game)
    print(f"Attackers: {attackers}")
    
    if not attackers:
        print("FAILED: Attackers list is empty! Fallback didn't work?")
        return False 
        
    print(f"SUCCESS: Attackers found via fallback: {attackers}")
    return True

if __name__ == "__main__":
    reproduced = reproduce()
    if reproduced:
        print("Issue Reproduced.")
        exit(0)
    else:
        print("Issue NOT Reproduced.")
        exit(1)
