
import numpy as np
import random
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color
from game3d.game.turnmove import make_move

def monitor_vectorsliders():
    game = GameState.from_startpos()
    
    # Verify initial count
    coords, types, colors = game.cache_manager.occupancy_cache.get_all_occupied_vectorized()
    vs_count = np.sum(types == PieceType.VECTORSLIDER)
    print(f"Initial VectorSlider count: {vs_count}")
    
    turn = 0
    max_turns = 100
    
    while turn < max_turns:
        # Check count
        coords, types, colors = game.cache_manager.occupancy_cache.get_all_occupied_vectorized()
        vs_count = np.sum(types == PieceType.VECTORSLIDER)
        
        if vs_count > 2: # 1 per side = 2 total
            print(f"FAIL: VectorSlider count increased to {vs_count} at turn {turn}!")
            
            # Find the new ones
            vs_indices = np.where(types == PieceType.VECTORSLIDER)[0]
            for idx in vs_indices:
                print(f"VS at {coords[idx]} (Color: {colors[idx]})")
                
            return
            
        # Make a random move
        moves = game.legal_moves
        if len(moves) == 0:
            print("Game over (no moves)")
            break
            
        move = moves[random.randint(0, len(moves) - 1)]
        game = make_move(game, move)
        turn += 1
        
    print("Finished without duplication.")

if __name__ == "__main__":
    monitor_vectorsliders()
