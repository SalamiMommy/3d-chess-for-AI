
import numpy as np
from game3d.game.gamestate import GameState
from game3d.common.shared_types import PieceType, COORD_DTYPE, SIZE
from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS

def test_piece_moves(piece_type, piece_name):
    print(f"Testing {piece_name}...")
    
    # Initialize game
    game = GameState.from_startpos()
    
    # Clear board
    game.cache_manager.occupancy_cache.clear()
    game.cache_manager.consolidated_aura_cache._buffed_squares.fill(False)
    
    # Place piece at center (4, 4, 4)
    center = np.array([4, 4, 4], dtype=COORD_DTYPE)
    piece_data = np.array([int(piece_type), int(game.color)], dtype=np.int8)
    game.cache_manager.occupancy_cache.set_position(center, piece_data)
    
    # 1. Test Unbuffed
    moves = game.legal_moves
    
    # Expected unbuffed count: 26 (King moves)
    # Note: Some pieces have extra moves (Mirror teleport, Swapper swap)
    # But here board is empty, so no swaps/teleports unless condition met.
    # Mirror teleports to  (4,4,4) which is self, so usually skipped.
    
    base_king_moves = 26
    
    print(f"  Unbuffed moves: {len(moves)}")
    
    if len(moves) < base_king_moves:
        print(f"  FAIL: Expected at least {base_king_moves} moves, got {len(moves)}")
        return False
        
    # 2. Test Buffed
    # Manually buff the square
    game.cache_manager.consolidated_aura_cache._buffed_squares[4, 4, 4] = True
    
    moves_buffed = game.legal_moves
    
    # Expected buffed count: 124 (5x5x5 - 1)
    base_buffed_moves = 124
    
    print(f"  Buffed moves: {len(moves_buffed)}")
    
    if len(moves_buffed) < base_buffed_moves:
        print(f"  FAIL: Expected at least {base_buffed_moves} moves, got {len(moves_buffed)}")
        return False
        
    print(f"  PASS")
    return True

def run_tests():
    pieces_to_test = [
        (PieceType.WALL, "Wall"),
        (PieceType.ARMOUR, "Armour"),
        (PieceType.MIRROR, "Mirror"),
        (PieceType.SWAPPER, "Swapper"),
        (PieceType.INFILTRATOR, "Infiltrator"),
        (PieceType.HIVE, "Hive"),
        (PieceType.FRIENDLYTELEPORTER, "FriendlyTP"),
        (PieceType.FREEZER, "Freezer"),
        (PieceType.BOMB, "Bomb"),
        (PieceType.ARCHER, "Archer")
    ]
    
    all_passed = True
    for p_type, p_name in pieces_to_test:
        if not test_piece_moves(p_type, p_name):
            all_passed = False
            
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")

if __name__ == "__main__":
    run_tests()
