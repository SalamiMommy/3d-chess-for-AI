
import numpy as np
from game3d.pieces.pieces.reflector import generate_reflecting_bishop_moves
from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType

def test_reflector_moves():
    print("Testing Reflector Moves...")
    
    # Create a dummy game state (we might need to mock cache manager if it's complex)
    # But generate_reflecting_bishop_moves takes cache_manager.
    # Let's try to set up a minimal environment.
    
    try:
        game = GameState.from_startpos()
        # Clear board
        game.cache_manager.occupancy_cache.clear()
    except Exception as e:
        print(f"Failed to init game state: {e}")
        return

    # Place a reflector in the center to see directions
    # Center of 9x9x9 is 4,4,4
    pos = np.array([4, 4, 4])
    
    # We need to mock the cache manager or use the real one.
    # The real one requires a game state.
    
    print(f"Generating moves for Reflector at {pos}")
    moves = generate_reflecting_bishop_moves(
        cache_manager=game.cache_manager,
        color=Color.WHITE,
        pos=pos,
        max_bounces=2, # Updated default
        ignore_occupancy=True
    )
    
    print(f"Total moves found: {len(moves)}")
    
    # Analyze directions
    directions = set()
    for move in moves:
        start = move[:3]
        end = move[3:]
        # This is a full path, but let's look at immediate neighbors to see directions
        # Actually, the moves returned are (start, end) pairs.
        # But wait, generate_reflecting_bishop_moves returns (start_x, start_y, start_z, end_x, end_y, end_z)
        
        # To find initial directions, we can look at moves that are 1 step away?
        # Or just infer from the endpoints.
        # Since it bounces, it's hard to infer initial direction from just endpoint.
        pass

    # Let's just count how many unique first steps there are.
    # We can trace rays manually or just look at the code.
    # The code uses _REFLECTOR_DIRS.
    
    # Let's place it near a wall to test bouncing.
    # Place at 1,1,1.
    pos_corner = np.array([0, 0, 0])
    print(f"\nGenerating moves for Reflector at {pos_corner}")
    moves_corner = generate_reflecting_bishop_moves(
        game.cache_manager,
        Color.WHITE,
        pos_corner,
        max_bounces=2,
        ignore_occupancy=True
    )
    print(f"Total moves from corner: {len(moves_corner)}")
    
    # Print some moves to verify bouncing
    # A move from 0,0,0 with direction 1,1,1 goes to 8,8,8 (if 9x9x9)
    # If it hits a wall, it should reflect.
    
    # Let's look for a bounce.
    # If we are at 0,0,0 and go +1,+1,+1, we hit 8,8,8.
    # If we are at 4,4,4 and go +1,+1,+1, we hit 8,8,8.
    
    # Let's try a specific bounce case.
    # Position: [7, 4, 4] (near x-edge)
    # Direction: [1, 0, 0] (not a valid reflector dir currently)
    # Current dirs are diagonals.
    
    # Current Reflector Dirs: (+-1, +-1, +-1) (8 dirs)
    # Bishop Dirs: (+-1, +-1, 0) and perms (12 dirs)
    
    # Let's just verify the number of directions by placing in center and counting immediate neighbors
    # (assuming no immediate blockage)
    
    # Filter moves that are distance 1 (Chebyshev distance = 1)
    immediate_moves = []
    for m in moves:
        start = m[:3]
        end = m[3:]
        diff = np.abs(end - start)
        if np.max(diff) == 1:
            immediate_moves.append(end - start)
            
    unique_dirs = np.unique(immediate_moves, axis=0)
    print(f"Unique initial directions found: {len(unique_dirs)}")
    print("Directions:")
    print(unique_dirs)
    
    if len(unique_dirs) == 12:
        print("SUCCESS: 12 unique directions found.")
    else:
        print(f"FAILURE: Expected 12 directions, found {len(unique_dirs)}.")

if __name__ == "__main__":
    test_reflector_moves()
