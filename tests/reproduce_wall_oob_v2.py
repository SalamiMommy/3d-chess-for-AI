
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.game.gamestate import GameState
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.pieces.pieces.wall import generate_wall_moves
from game3d.game.turnmove import make_move, apply_forced_moves
from game3d.common.coord_utils import coords_to_keys

def test_wall_move_generation():
    print("Testing Wall Move Generation near edge...")
    from game3d.board.board import Board
    board = Board.startpos()
    game = GameState(board=board)
    
    # Place a Wall at [7, 0, 0] (Valid anchor, occupies 7,0,0; 8,0,0; 7,1,0; 8,1,0)
    # If it moves +1 in X, it goes to [8, 0, 0], which occupies 8,0,0; 9,0,0... OOB!
    wall_pos = np.array([7, 0, 0])
    game.cache_manager.occupancy_cache.set_position(wall_pos, np.array([PieceType.WALL, Color.WHITE]))
    # We must also set the other parts of the wall for it to be a valid wall?
    # wall.py checks neighbors to identify anchors.
    # "A wall piece is an anchor ONLY if it has no wall to its left (x-1) and no wall above (y-1)"
    # So we just need to make sure [6,0,0] and [7,-1,0] are NOT walls.
    
    # Let's populate the full 2x2 block just to be safe and realistic
    game.cache_manager.occupancy_cache.set_position(np.array([8, 0, 0]), np.array([PieceType.WALL, Color.WHITE]))
    game.cache_manager.occupancy_cache.set_position(np.array([7, 1, 0]), np.array([PieceType.WALL, Color.WHITE]))
    game.cache_manager.occupancy_cache.set_position(np.array([8, 1, 0]), np.array([PieceType.WALL, Color.WHITE]))
    
    moves = generate_wall_moves(game.cache_manager, Color.WHITE, wall_pos)
    print(f"Generated {len(moves)} moves for Wall at {wall_pos}")
    
    for move in moves:
        dest = move[3:]
        print(f"Move to {dest}")
        if dest[0] >= SIZE - 1:
            print(f"❌ ERROR: Generated move to {dest} which is too close to edge (SIZE={SIZE})")
            
            # Try to execute it to see if it crashes
            try:
                make_move(game, move)
            except ValueError as e:
                print(f"Caught expected error during execution: {e}")
            except Exception as e:
                print(f"Caught UNEXPECTED error: {e}")

def test_whitehole_push_wall():
    print("\nTesting Whitehole Pushing Wall...")
    from game3d.board.board import Board
    board = Board.startpos()
    game = GameState(board=board)
    
    # Place Wall at [6, 0, 0]
    wall_pos = np.array([6, 0, 0])
    # Populate 2x2
    for dx in [0, 1]:
        for dy in [0, 1]:
            pos = wall_pos + np.array([dx, dy, 0])
            game.cache_manager.occupancy_cache.set_position(pos, np.array([PieceType.WALL, Color.WHITE]))
            
    # Place Whitehole at [5, 0, 0] (Should push Wall to [7, 0, 0])
    # Wait, Whitehole pushes radially.
    # If Whitehole is at [5, 0, 0], and Wall part is at [6, 0, 0], direction is +1 X.
    # Pushes [6, 0, 0] to [7, 0, 0].
    # But what about [7, 0, 0] (the other part of the wall)? It is at dist 2 from Whitehole.
    # Whitehole radius is usually larger?
    
    wh_pos = np.array([5, 0, 0])
    game.cache_manager.occupancy_cache.set_position(wh_pos, np.array([PieceType.WHITEHOLE, Color.BLACK]))
    
    # Manually trigger forced moves
    from game3d.pieces.pieces.whitehole import push_candidates_vectorized
    forced_moves = push_candidates_vectorized(game.cache_manager, Color.BLACK)
    print(f"Generated {len(forced_moves)} forced moves")
    for fm in forced_moves:
        print(f"Forced Move: {fm[:3]} -> {fm[3:]}")
        
    if len(forced_moves) > 0:
        try:
            apply_forced_moves(game, forced_moves)
            print("Applied forced moves successfully")
        except ValueError as e:
            print(f"❌ ERROR applying forced moves: {e}")

def test_swapper_swap_wall():
    print("\nTesting Swapper Swapping with Wall near edge...")
    from game3d.board.board import Board
    board = Board.startpos()
    game = GameState(board=board)
    
    # Place Wall at [7, 6, 6] (Safe anchor, occupies 7,6,6; 8,6,6; 7,7,6; 8,7,6)
    wall_pos = np.array([7, 6, 6])
    game.cache_manager.occupancy_cache.set_position(wall_pos, np.array([PieceType.WALL, Color.WHITE]))
    
    # Place Swapper at [8, 6, 6] (Valid for Swapper, invalid for Wall anchor)
    # Wait, if Wall is at [7, 6, 6], it occupies [8, 6, 6].
    # So Swapper cannot be at [8, 6, 6] (collision).
    # Let's place Swapper at [6, 6, 6] (Valid).
    # If it swaps, Wall goes to [6, 6, 6] (Valid).
    # We need Wall to go to an INVALID anchor.
    # Wall anchor must be < 8.
    # If Wall moves to [8, 6, 6], it is invalid.
    # So Swapper must be at [8, 6, 6].
    # But Wall cannot be at [7, 6, 6] because it occupies [8, 6, 6].
    # Wall at [5, 5, 5] occupies [5,5,5]...[6,6,5]. Far away.
    # Swapper at [8, 6, 6].
    # Swap -> Wall at [8, 6, 6]. Invalid.
    
    # So why did the previous test not generate the move?
    # Maybe Swapper range?
    pass
    
    # Generate Swapper moves
    from game3d.pieces.pieces.swapper import generate_swapper_moves
    moves = generate_swapper_moves(game.cache_manager, Color.WHITE, swapper_pos)
    print(f"Generated {len(moves)} moves for Swapper at {swapper_pos}")
    
    for move in moves:
        dest = move[3:]
        # Check if it swaps with Wall
        if np.array_equal(dest, wall_pos):
            print(f"Found swap move with Wall at {wall_pos}")
            try:
                make_move(game, move)
                print("❌ Executed swap move successfully (Should have failed!)")
            except ValueError as e:
                print(f"✅ Caught expected error: {e}")
            except Exception as e:
                print(f"Caught UNEXPECTED error: {e}")

if __name__ == "__main__":
    try:
        test_wall_move_generation()
        test_whitehole_push_wall()
        test_swapper_swap_wall()
    except Exception as e:
        print(f"Test failed with: {e}")
        import traceback
        traceback.print_exc()
