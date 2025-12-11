"""
Test if frozen pieces are causing the Unknown attackers bug.
Hypothesis: Frozen pieces should still generate attack threats for check detection,
but the current generate_pseudolegal_moves skips frozen pieces entirely.
"""
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from game3d.board.board import Board
from game3d.game.gamestate import GameState
from game3d.cache.manager import OptimizedCacheManager
from game3d.common.shared_types import (
    PieceType, Color, COORD_DTYPE, SIZE,
    COLOR_WHITE, COLOR_BLACK, MinimalStateProxy
)
from game3d.core.buffer import state_to_buffer
from game3d.core.api import generate_pseudolegal_moves
from game3d.attacks.check import _find_attackers_of_square, get_attackers


def setup_frozen_attacker_scenario():
    """
    Create a scenario where:
    - Black King at (4,4,8)
    - White Rook at (4,0,8) - CAN attack Black King along y-axis
    - Black Freezer at (5,0,8) - Freezes the White Rook
    - White King at (4,4,0) - safe
    
    The Rook is frozen by the Freezer, so it can't MOVE.
    BUT it should still COUNT as attacking the Black King for check purposes!
    """
    print("=== Setup Frozen Attacker Scenario ===")
    
    # Create board
    board = Board.startpos()
    game = GameState(board, Color.BLACK.value)  # BLACK to move
    
    occ_cache = game.cache_manager.occupancy_cache
    
    # Clear board
    coords, _, _ = occ_cache.get_all_occupied_vectorized()
    for c in coords:
        occ_cache.set_position(c, None)
    
    # Setup pieces
    # Black King at (4,4,8)
    b_king_pos = np.array([4, 4, 8], dtype=np.int16)
    occ_cache.set_position(b_king_pos, np.array([PieceType.KING.value, Color.BLACK.value]))
    
    # White Rook at (4,0,8) - can attack along y-axis to (4,4,8)
    w_rook_pos = np.array([4, 0, 8], dtype=np.int16)
    occ_cache.set_position(w_rook_pos, np.array([PieceType.ROOK.value, Color.WHITE.value]))
    
    # Black Freezer at (5,0,8) - within range 2 of the Rook
    # This should freeze the Rook since Freezer freezes ENEMY pieces in radius 2
    b_freezer_pos = np.array([5, 0, 8], dtype=np.int16)
    occ_cache.set_position(b_freezer_pos, np.array([PieceType.FREEZER.value, Color.BLACK.value]))
    
    # White King at (4,4,0) - needed for valid game
    w_king_pos = np.array([4, 4, 0], dtype=np.int16)
    occ_cache.set_position(w_king_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
    
    # Invalidate move cache
    game.cache_manager.move_cache.invalidate_pseudolegal_moves(None)
    
    # Rebuild aura cache
    game.cache_manager.aura_cache.rebuild_from_board(occ_cache)
    
    # Verify setup
    print("\nBoard state:")
    coords, types, colors = occ_cache.get_all_occupied_vectorized()
    for i in range(len(coords)):
        c = coords[i]
        t = types[i]
        col = colors[i]
        color_name = Color(col).name
        type_name = PieceType(t).name
        print(f"  ({c[0]},{c[1]},{c[2]}): {type_name} ({color_name})")
    
    return game


def trace_get_attackers(game):
    """Trace the get_attackers logic step by step."""
    print("\n=== Trace get_attackers ===")
    
    cache = game.cache_manager
    occ_cache = cache.occupancy_cache
    
    # Step 1: Identify colors
    king_color = game.color
    opponent_color = Color(king_color).opposite().value
    print(f"king_color: {Color(king_color).name}")
    print(f"opponent_color: {Color(opponent_color).name}")
    
    # Step 2: Find king
    king_pos = occ_cache.find_king(king_color)
    print(f"King position: {king_pos}")
    
    # Step 3: Create buffer for opponent
    dummy_state = MinimalStateProxy(game.board, opponent_color, cache)
    print(f"\nMinimalStateProxy.color = {dummy_state.color}")
    
    buffer = state_to_buffer(dummy_state, readonly=True)
    print(f"Buffer meta[0] (active_color) = {buffer.meta[0]}")
    print(f"Buffer occupied_count = {buffer.occupied_count}")
    
    # Check aura maps
    print("\nAura maps for WHITE pieces:")
    print(f"  is_frozen at Rook (4,0,8): {buffer.is_frozen[4, 0, 8]}")
    print(f"  is_debuffed at Rook (4,0,8): {buffer.is_debuffed[4, 0, 8]}")
    print(f"  is_buffed at Rook (4,0,8): {buffer.is_buffed[4, 0, 8]}")
    
    # Check what pieces the buffer sees
    print("\nPieces in buffer:")
    for i in range(buffer.occupied_count):
        c = buffer.occupied_coords[i]
        t = buffer.occupied_types[i]
        col = buffer.occupied_colors[i]
        frozen = buffer.is_frozen[c[0], c[1], c[2]]
        print(f"  [{i}] ({c[0]},{c[1]},{c[2]}): type={t}, color={col}, frozen={frozen}")
    
    # Step 4: Generate moves
    opponent_moves = generate_pseudolegal_moves(buffer)
    print(f"\nGenerated {opponent_moves.shape[0]} moves (WHITE moves)")
    
    if opponent_moves.shape[0] > 0:
        # Show unique source positions
        unique_sources = np.unique(opponent_moves[:, :3], axis=0)
        print(f"Unique source positions: {len(unique_sources)}")
        for src in unique_sources:
            print(f"  ({src[0]},{src[1]},{src[2]})")
    else:
        print("NO MOVES GENERATED!")
    
    # Step 5: Find attackers
    king_pos_arr = king_pos.astype(COORD_DTYPE)
    attacker_indices = _find_attackers_of_square(king_pos_arr, opponent_moves, opponent_moves[:, :3])
    print(f"\nAttacker indices: {attacker_indices}")
    
    return len(attacker_indices) > 0


def main():
    game = setup_frozen_attacker_scenario()
    found = trace_get_attackers(game)
    
    # Now call the actual get_attackers
    print("\n=== Calling actual get_attackers ===")
    attackers = get_attackers(game)
    print(f"Result: {attackers}")
    
    if not found and not attackers:
        print("\n*** BUG REPRODUCED! ***")
        print("The Rook at (4,0,8) is FROZEN by the Black Freezer.")
        print("Therefore generate_pseudolegal_moves skips it.")
        print("But the Rook STILL attacks the Black King!")
        print("This is the cause of 'Attackers: Unknown' in checkmate logs.")
    else:
        print("\n*** BUG NOT REPRODUCED ***")


if __name__ == "__main__":
    main()
