
import numpy as np
import random
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color
from game3d.game.turnmove import make_move

def reproduce_v2():
    game = GameState.from_startpos()
    
    # Target squares: (3,2,0) and (5,6,1)
    # Initially:
    # (3,2,0) -> Hive (11)
    # (5,6,1) -> Rook (4)
    
    targets = [
        (3, 2, 0, PieceType.HIVE),
        (5, 6, 1, PieceType.ROOK)
    ]
    
    print("Starting reproduction run...")
    
    for turn in range(200):
        # Check targets
        for x, y, z, expected_type in targets:
            piece = game.cache_manager.occupancy_cache.get(np.array([x, y, z]))
            if piece:
                actual_type = piece["piece_type"]
                if actual_type == PieceType.VECTORSLIDER and expected_type != PieceType.VECTORSLIDER:
                    print(f"FAIL: Piece at ({x},{y},{z}) became VECTORSLIDER at turn {turn}!")
                    return
                
                # Update expected type if it changed (e.g. captured and replaced)
                # But if it changed to VectorSlider, we caught it.
            else:
                # Piece moved or captured
                pass
                
        # Make random move
        moves = game.legal_moves
        if len(moves) == 0:
            break
        move = moves[random.randint(0, len(moves) - 1)]
        game = make_move(game, move)
        
    print("Finished without reproduction.")

if __name__ == "__main__":
    reproduce_v2()
