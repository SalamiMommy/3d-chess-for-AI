
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType
from game3d.game.turnmove import legal_moves
from game3d.cache.manager import OptimizedCacheManager

def test_king_capture_filtering():
    print("=== TEST 1: With Priest ===")
    print("Initializing Game State...")
    board = Board.empty()
    
    # Setup pieces
    # White King at (0, 0, 0)
    board.set_piece_at(np.array([0, 0, 0]), PieceType.KING, Color.WHITE)
    
    # Black King at (7, 7, 7)
    black_king_pos = np.array([7, 7, 7])
    board.set_piece_at(black_king_pos, PieceType.KING, Color.BLACK)
    
    # Black Priest at (7, 7, 6)
    black_priest_pos = np.array([7, 7, 5])
    board.set_piece_at(black_priest_pos, PieceType.PRIEST, Color.BLACK)
    
    # White Queen adjacent to Black King
    white_queen_pos = np.array([7, 7, 6])
    board.set_piece_at(white_queen_pos, PieceType.QUEEN, Color.WHITE)
    
    cache_manager = OptimizedCacheManager(board)
    game_state = GameState(board, Color.WHITE, cache_manager)
    
    print(f"White has priest: {cache_manager.occupancy_cache.has_priest(Color.WHITE)}")
    print(f"Black has priest: {cache_manager.occupancy_cache.has_priest(Color.BLACK)}")
    print("Generating moves with Black Priest present...")
    moves = legal_moves(game_state)
    print(f"Total moves for White: {len(moves)}")
    
    # Check if capture of (7, 7, 7) is in moves
    capture_found = False
    for move in moves:
        # Structured array: extract fields
        to_coord = np.array([move['to_x'], move['to_y'], move['to_z']])
        if np.array_equal(to_coord, black_king_pos):
            capture_found = True
            break
            
    if capture_found:
        print("❌ FAILURE: King capture found despite Priest presence!")
        return False
    else:
        print("✅ SUCCESS: King capture filtered out due to Priest.")
        
    # TEST 2: Without priest
    print("\n=== TEST 2: Without Priest ===")
    print("Initializing new Game State without Priest...")
    board2 = Board.empty()
    
    # White King at (0, 0, 0)
    board2.set_piece_at(np.array([0, 0, 0]), PieceType.KING, Color.WHITE)
    
    # Black King at (7, 7, 7) - NO PRIEST
    black_king_pos = np.array([7, 7, 7])
    board2.set_piece_at(black_king_pos, PieceType.KING, Color.BLACK)
    
    # White Queen adjacent to Black King - can definitely capture it
    white_queen_pos = np.array([7, 7, 6])
    board2.set_piece_at(white_queen_pos, PieceType.QUEEN, Color.WHITE)
    
    cache_manager2 = OptimizedCacheManager(board2)
    game_state2 = GameState(board2, Color.WHITE, cache_manager2)
    
    # Debug: Check all pieces on the board
    print("\nPieces on board:")
    coords, types, colors = cache_manager2.occupancy_cache.get_all_occupied_vectorized()
    for i in range(len(coords)):
        print(f"  {coords[i]} - Type: {types[i]}, Color: {colors[i]}")
    
    print(f"\nBlack has priest: {cache_manager2.occupancy_cache.has_priest(Color.BLACK)}")
    print("Generating moves without Black Priest...")
    moves2 = legal_moves(game_state2)
    print(f"Total moves generated: {len(moves2)}")
    
    # Debug: Print all moves by piece
    print("\nAll moves by piece:")
    for i, move in enumerate(moves2[:10]):  # Just first 10 for brevity
        print(f"  Move {i}: {move}")
        print(f"    from_x={move['from_x']}, from_y={move['from_y']}, from_z={move['from_z']}")
        print(f"    to_x={move['to_x']}, to_y={move['to_y']}, to_z={move['to_z']}")
    
    capture_found = False
    for move in moves2:
        to_x, to_y, to_z = move['to_x'], move['to_y'], move['to_z']
        if to_x == black_king_pos[0] and to_y == black_king_pos[1] and to_z == black_king_pos[2]:
            print(f"\nFound king capture: {move}")
            capture_found = True
            break
            
    if capture_found:
        print("✅ SUCCESS: King capture allowed without Priest.")
        return True
    else:
        print("❌ FAILURE: King capture NOT found even without Priest!")
        return False

if __name__ == "__main__":
    test_king_capture_filtering()
