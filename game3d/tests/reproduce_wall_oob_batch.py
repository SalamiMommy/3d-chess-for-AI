"""Reproduce Wall out-of-bounds coordinate bug."""
import numpy as np
import sys
sys.path.insert(0, '/home/salamimommy/Documents/code/3d-chess-for-AI')

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, Color, SIZE, COORD_DTYPE
from game3d.pieces.pieces.wall import generate_wall_moves

def test_wall_edge_cases():
    """Test Wall move generation at board edges."""
    print(f"Board size: {SIZE}, Valid anchor range: [0, {SIZE-2}]")
    print("=" * 80)
    
    # Create game state
    board = Board.empty()
    state = GameState(board)
    
    # Test cases: Wall anchors near board edges
    test_cases = [
        # (anchor, description)
        ([8, 7, 7], "Anchor at (8,7,7) - x at max"),
        ([7, 8, 7], "Anchor at (7,8,7) - y at max"),
        ([7, 7, 7], "Anchor at (7,7,7) - both at max-1"),
        ([8, 6, 7], "Anchor at (8,6,7) - x at max, y valid"),
        ([6, 8, 3], "Anchor at (6,8,3) - y at max, x valid"),
    ]
    
    errors = []
    
    for anchor_list, desc in test_cases:
        anchor = np.array(anchor_list, dtype=COORD_DTYPE)
        print(f"\n{desc}")
        print(f"  Anchor: {anchor}")
        
        # Place a white Wall at this anchor
        state.cache_manager.occupancy_cache.set_position(anchor, {
            'piece_type': PieceType.WALL,
            'color': Color.WHITE
        })
        
        # Generate moves (unbuffed)
        moves = generate_wall_moves(state.cache_manager, Color.WHITE, anchor)
        
        print(f"  Generated {len(moves)} moves")
        
        # Validate each move
        for i, move in enumerate(moves):
            src = move[:3]
            dest = move[3:]
            
            # Check destination anchor validity
            if dest[0] >= SIZE - 1 or dest[1] >= SIZE - 1 or dest[2] >= SIZE:
                error_msg = f"    ❌ Move {i}: {src} -> {dest} - Dest anchor INVALID (x={dest[0]}, y={dest[1]}, z={dest[2]})"
                print(error_msg)
                errors.append(error_msg)
                continue
            
            # Check all 4 destination squares
            block_offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=COORD_DTYPE)
            dest_squares = dest + block_offsets
            
            for j, sq in enumerate(dest_squares):
                if sq[0] >= SIZE or sq[1] >= SIZE or sq[2] >= SIZE or sq[0] < 0 or sq[1] < 0 or sq[2] < 0:
                    error_msg = f"    ❌ Move {i}: {src} -> {dest} - Block square {j} OOB: {sq}"
                    print(error_msg)
                    errors.append(error_msg)
                    break
            else:
                print(f"    ✅ Move {i}: {src} -> {dest} - Valid")
        
        # Clear for next test
        state.cache_manager.occupancy_cache.set_position(anchor, None)
    
    print("\n" + "=" * 80)
    if errors:
        print(f"❌ FOUND {len(errors)} ERRORS:")
        for err in errors:
            print(err)
        return False
    else:
        print("✅ ALL MOVES VALID")
        return True

if __name__ == '__main__':
    success = test_wall_edge_cases()
    sys.exit(0 if success else 1)
