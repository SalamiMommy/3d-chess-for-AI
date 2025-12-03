
import numpy as np
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, SIZE
from game3d.game.terminal import is_game_over
from game3d.game.turnmove import legal_moves

def reproduce_regression():
    print("Reproducing Regression...")
    
    # Initialize GameState with empty board
    board = Board()
    game_state = GameState(board, Color.WHITE)
    
    # Define pieces for checkmate scenario
    coords = [
        [0,0,0], # White King
        [0,0,5], # Black Rook
        [0,5,0], # Black Rook
        [5,0,0], # Black Rook
        [0,0,2], # Black Queen
        [0,2,0], # Black Queen
        [2,0,0], # Black Queen
        [2,2,2], # Black Queen
        [2,2,0], # Black Queen
        [2,0,2], # Black Queen
        [0,2,2], # Black Queen
        [1,1,5]  # Black Rook (Attacks [1,1,1])
    ]
    
    types = [
        PieceType.KING,
        PieceType.ROOK,
        PieceType.ROOK,
        PieceType.ROOK,
        PieceType.QUEEN,
        PieceType.QUEEN,
        PieceType.QUEEN,
        PieceType.QUEEN,
        PieceType.QUEEN,
        PieceType.QUEEN,
        PieceType.QUEEN,
        PieceType.ROOK
    ]
    
    colors = [
        Color.WHITE,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK,
        Color.BLACK
    ]
    
    # Convert to numpy arrays
    coords_arr = np.array(coords, dtype=np.int32)
    types_arr = np.array([t.value for t in types], dtype=np.int8)
    colors_arr = np.array(colors, dtype=np.int8)
    
    # Rebuild cache with this state
    if game_state.cache_manager:
        game_state.cache_manager.occupancy_cache.rebuild(coords_arr, types_arr, colors_arr)
        
    # Check legal moves
    moves = legal_moves(game_state)
    print(f"Legal moves count: {len(moves)}")
    for move in moves:
        print(f"Move: {move}")
        
    # Check if game over
    result = is_game_over(game_state)
    print(f"Is Game Over: {result}")

    # DEBUG: Check if [1, 1, 1] is attacked
    print("\n--- Debugging [1, 1, 1] ---")
    target = np.array([1, 1, 1], dtype=np.int16)
    from game3d.attacks.check import square_attacked_by
    is_attacked = square_attacked_by(game_state.board, game_state.color, target, Color.BLACK.value, game_state.cache_manager)
    print(f"Is [1, 1, 1] attacked by BLACK? {is_attacked}")
    
    # Check Queen at [2, 2, 2]
    print("Checking Queen at [2, 2, 2]:")
    queen_pos = np.array([2, 2, 2], dtype=np.int16)
    queen_piece = game_state.cache_manager.occupancy_cache.get(queen_pos)
    print(f"Piece at [2, 2, 2]: {queen_piece}")
    
    # Check if Queen can move to [1, 1, 1]
    from game3d.movement.generator import generate_legal_moves_for_piece
    queen_moves = generate_legal_moves_for_piece(game_state, queen_pos)
    can_hit_target = False
    for m in queen_moves:
        if np.array_equal(m[3:6], target):
            can_hit_target = True
            break
    print(f"Can Queen at [2, 2, 2] move to [1, 1, 1]? {can_hit_target}")


if __name__ == "__main__":
    reproduce_regression()
