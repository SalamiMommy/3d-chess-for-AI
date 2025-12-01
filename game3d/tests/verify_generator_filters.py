
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, PIECE_TYPE_DTYPE
from game3d.movement.generator import generate_legal_moves

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pinned_pieces():
    print("\n=== Testing Pinned Pieces Filter ===")
    
    # 1. Setup Board
    board = Board()
    state = GameState(board, color=Color.WHITE)
    state.cache_manager.occupancy_cache.clear()
    
    # 2. Place Pieces
    king_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    rook_pos = np.array([0, 1, 0], dtype=COORD_DTYPE)
    enemy_rook_pos = np.array([0, 5, 0], dtype=COORD_DTYPE)
    
    state.cache_manager.occupancy_cache.set_position(king_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
    state.cache_manager.occupancy_cache.set_position(rook_pos, np.array([PieceType.ROOK.value, Color.WHITE.value]))
    state.cache_manager.occupancy_cache.set_position(enemy_rook_pos, np.array([PieceType.ROOK.value, Color.BLACK.value]))
    
    # 2b. Generate Enemy Moves (to populate raw moves for pin detection)
    # We need to temporarily switch color or manually trigger generation
    original_color = state.color
    state.color = Color.BLACK
    _ = generate_legal_moves(state) # This populates Black's moves in cache
    state.color = original_color
    
    # 3. Generate Moves
    moves = generate_legal_moves(state)
    
    # 4. Verify Rook Moves
    # Rook is at [0, 1, 0]. Pinned by [0, 5, 0] against King at [0, 0, 0].
    # Allowed moves: [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0] (capture)
    
    rook_moves = moves[(moves[:, 0] == 0) & (moves[:, 1] == 1) & (moves[:, 2] == 0)]
    
    print(f"Generated {len(rook_moves)} moves for pinned Rook.")
    
    expected_destinations = {
        (0, 2, 0), (0, 3, 0), (0, 4, 0), (0, 5, 0)
    }
    
    generated_destinations = set()
    for move in rook_moves:
        dest = tuple(move[3:])
        generated_destinations.add(dest)
        
    print(f"Expected: {expected_destinations}")
    print(f"Got: {generated_destinations}")
    
    if expected_destinations == generated_destinations:
        print("✅ Pinned piece filtering correct.")
    else:
        print("❌ Pinned piece filtering FAILED.")
        # Check if it generated illegal moves (e.g. side moves)
        diff = generated_destinations - expected_destinations
        if diff:
            print(f"Illegal moves generated: {diff}")

def test_trailblazer_filter():
    print("\n=== Testing Trailblazer Filter ===")
    
    # 1. Setup Board
    board = Board()
    state = GameState(board, color=Color.WHITE)
    state.cache_manager.occupancy_cache.clear()
    state.cache_manager.trailblaze_cache.clear()
    
    # 2. Place King and Enemy Trailblazer
    king_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    enemy_tb_pos = np.array([4, 4, 0], dtype=COORD_DTYPE)
    
    state.cache_manager.occupancy_cache.set_position(king_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
    state.cache_manager.occupancy_cache.set_position(enemy_tb_pos, np.array([PieceType.TRAILBLAZER.value, Color.BLACK.value]))
    
    # 3. Create a Trail
    # Add a trail at [5, 5, 5] (one of King's potential moves)
    trail_pos = np.array([[5, 5, 5]], dtype=COORD_DTYPE)
    state.cache_manager.trailblaze_cache.add_trail(enemy_tb_pos, trail_pos, Color.BLACK)
    
    # 4. Set King Counters to 2 (Danger Zone)
    # We need to manually set the counter for the King's position?
    # No, the filter checks `batch_get_counters(king_positions)`.
    # So we need to set the counter at [4, 4, 4] to >= 2.
    
    # Increment counter twice
    state.cache_manager.trailblaze_cache.increment_counter(king_pos)
    state.cache_manager.trailblaze_cache.increment_counter(king_pos)
    
    print(f"King counter: {state.cache_manager.trailblaze_cache.get_counter(king_pos)}")
    
    # 5. Generate Moves
    moves = generate_legal_moves(state)
    
    # 6. Verify King Moves
    # King should NOT be able to move to [5, 5, 5] because it hits a trail and counter >= 2.
    
    king_moves = moves[(moves[:, 0] == 4) & (moves[:, 1] == 4) & (moves[:, 2] == 4)]
    
    destinations = [tuple(m[3:]) for m in king_moves]
    
    if (5, 5, 5) in destinations:
        print("❌ Trailblazer filtering FAILED. King moved to trail square [5, 5, 5].")
    else:
        print("✅ Trailblazer filtering correct. King avoided trail square.")
        
    # Verify King can move elsewhere (e.g. [4, 4, 5])
    if (4, 4, 5) in destinations:
        print("✅ King can move to safe squares.")
    else:
        print("⚠️ King cannot move to safe squares (might be blocked or other issue).")

if __name__ == "__main__":
    test_pinned_pieces()
    test_trailblazer_filter()
