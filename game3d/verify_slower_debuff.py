
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
from game3d.movement.generator import generate_legal_moves_for_piece

def verify_slower_debuff():
    print("Verifying Slower Piece Debuff...")
    
    # Initialize game state
    state = GameState()
    state.board.clear()
    
    # Setup:
    # Black Slower at (4, 4, 4)
    # White Queen at (4, 4, 5) (Inside aura, dist=1)
    # White Rook at (0, 0, 0) (Outside aura)
    
    slower_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    queen_pos = np.array([4, 4, 5], dtype=COORD_DTYPE)
    rook_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    
    state.board.place_piece(slower_pos, Color.BLACK, PieceType.SLOWER)
    state.board.place_piece(queen_pos, Color.WHITE, PieceType.QUEEN)
    state.board.place_piece(rook_pos, Color.WHITE, PieceType.ROOK)
    
    # Rebuild cache to ensure everything is up to date
    state.cache_manager._rebuild_occupancy_from_board()
    state.cache_manager.move_cache.invalidate()
    state.cache_manager.consolidated_aura_cache.invalidate_all()
    
    # Force aura cache update
    # We need to trigger occupancy change notification manually if we bypassed standard add_piece
    # But _rebuild_occupancy_from_board should handle it? 
    # Actually, _rebuild_occupancy_from_board calls occupancy_cache.rebuild, which might NOT notify listeners?
    # Let's check manager.py... _rebuild_occupancy_from_board calls self.occupancy_cache.rebuild
    # OccupancyCache.rebuild usually notifies listeners.
    # To be safe, let's manually trigger a notification or just rely on generate() triggering updates?
    # AuraCache updates on occupancy change.
    
    # Let's just run generation, it should handle everything.
    
    print("Generating moves for White...")
    state.color = Color.WHITE
    
    # 1. Check Queen moves (should be Pawn moves)
    queen_moves = generate_legal_moves_for_piece(state, queen_pos)
    print(f"Queen moves count: {len(queen_moves)}")
    
    # Expected Pawn moves for White at (4, 4, 5):
    # Push: (4, 5, 5)
    # Attacks: (3, 5, 4), (5, 5, 4), (3, 5, 6), (5, 5, 6) (if enemies present)
    # Since no enemies, only push should be available?
    # Wait, Pawn attack moves are only generated if there are capture targets?
    # Let's check pawn.py.
    # Yes, cap_moves are generated, but filtered?
    # "Capture moves ... allow_capture=True"
    # "Filter invalid captures... if cap_moves.size > 0... check victims"
    # So if no victims, no capture moves.
    
    # So Queen should only have 1 move: (4, 5, 5)
    
    expected_queen_dest = np.array([4, 5, 5], dtype=COORD_DTYPE)
    
    if len(queen_moves) != 1:
        print(f"FAILURE: Queen should have 1 move (Pawn push), got {len(queen_moves)}")
        for m in queen_moves:
            print(f"  {m[:3]} -> {m[3:]}")
    else:
        dest = queen_moves[0, 3:]
        if np.array_equal(dest, expected_queen_dest):
            print("SUCCESS: Queen has correct Pawn push move.")
        else:
            print(f"FAILURE: Queen move destination mismatch. Expected {expected_queen_dest}, got {dest}")

    # 2. Check Rook moves (should be normal Rook moves)
    rook_moves = generate_legal_moves_for_piece(state, rook_pos)
    print(f"Rook moves count: {len(rook_moves)}")
    
    # Rook at (0,0,0) on empty board has 3 * (SIZE-1) moves = 3 * 7 = 21 moves
    # (Assuming SIZE=8)
    # Let's just verify it has > 1 moves and includes (0, 1, 0)
    
    if len(rook_moves) > 1:
        print("SUCCESS: Rook has multiple moves (unaffected).")
    else:
        print(f"FAILURE: Rook should have many moves, got {len(rook_moves)}")

    # 3. Verify Slower moves (Black)
    print("\nSwitching to Black...")
    state.color = Color.BLACK
    slower_moves = generate_legal_moves_for_piece(state, slower_pos)
    print(f"Slower moves count: {len(slower_moves)}")
    
    # Slower moves like a King (26 directions)
    # At (4,4,4), it should have 26 moves (assuming no blocks)
    if len(slower_moves) == 26:
        print("SUCCESS: Slower has 26 moves (King-like).")
    else:
        print(f"FAILURE: Slower should have 26 moves, got {len(slower_moves)}")

if __name__ == "__main__":
    verify_slower_debuff()
