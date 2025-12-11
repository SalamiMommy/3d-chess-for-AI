
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.movement.generator import generate_legal_moves
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.attacks.check import king_in_check
from game3d.cache.caches.auracache import AuraCache

def reproduce():
    board = Board.startpos()
    state = GameState(board, Color.WHITE)
    occ = state.cache_manager.occupancy_cache
    
    # Clear board
    coords, _, _ = occ.get_all_occupied_vectorized()
    occ.batch_set_positions(coords, np.zeros((len(coords), 2), dtype=np.int32))
    
    # Setup Frozen Attacker Scenario:
    # White King at (0,0,0)
    # Black Rook at (0,0,5) -> Attacks (0,0,0)
    # Black Rook is FROZEN (e.g. by Geomancer aura or just set manually)
    
    occ.set_position(np.array([0,0,0]), np.array([PieceType.KING, Color.WHITE]))
    pos_rook = np.array([0,0,5])
    occ.set_position(pos_rook, np.array([PieceType.ROOK, Color.BLACK]))
    
    # Manually Freeze the Rook
    # We can use consolidated_aura_cache to set freeze
    aura = state.cache_manager.consolidated_aura_cache
    # Freeze logic usually works by turn number. 
    # Let's mock the freeze check or inject a freeze.
    # The AuraCache has 'trigger_freeze' or 'add_aura'.
    # Easier: Mock the `batch_is_frozen` method or manually set internal state if accessible.
    
    # Actually, let's use the public API if possible.
    # Geomancer blocks coords.
    # cache_manager.geomancy_cache.block_coords(coords, expiry)
    
    # But checking generator.py: 
    # is_frozen = state.cache_manager.consolidated_aura_cache.batch_is_frozen(...)
    
    # Let's inspect AuraCache briefly or try to inject data.
    # For now, I will try to use the cache manager methods.
    # If I can't easily freeze, I'll mock it.
    
    # Rebuild state logic
    state._zkey = state.cache_manager._compute_initial_zobrist(state.color)
    state.cache_manager.move_cache.invalidate()
    
    # Force 'batch_is_frozen' to return True for the Rook
    original_batch_is_frozen = aura.batch_is_frozen
    
    def mock_batch_is_frozen(coords, current_turn, my_color):
        # Return True for Rook pos
        is_froz = np.zeros(len(coords), dtype=bool)
        for i, c in enumerate(coords):
            if np.array_equal(c, pos_rook):
                is_froz[i] = True
        return is_froz
        
    aura.batch_is_frozen = mock_batch_is_frozen
    
    # Force full cache invalidation to prevent incremental crash
    state.cache_manager.move_cache.invalidate()

    
    # Regenerate moves for Black (Attacker)
    # This should update the pseudolegal cache
    # We need to simulate the turn flow where Black's moves are generated
    
    # Switch to Black to refresh their cache
    state.color = Color.BLACK
    import game3d.movement.generator as gen_module
    from game3d.movement.generator import initialize_generator
    if gen_module._generator is None: initialize_generator()
    gen_module._generator.refresh_pseudolegal_moves(state)
    
    # Now switch back to White (Victim)
    state.color = Color.WHITE
    
    # Check if King is in check
    # If Frozen Rook attacks, it should be True.
    # If bug exists, it will be False.
    in_check = king_in_check(state.board, state.color, state.color, state.cache_manager)
    print(f"Is White in check (Attacker Frozen)? {in_check}")
    
    if not in_check:
        print("BUG REPRODUCED: Frozen piece does not give check!")
        return True
    else:
        print("Bug not reproduced (Frozen piece gives check)")
        return False

if __name__ == "__main__":
    if reproduce():
        exit(0)
    else:
        exit(1)
