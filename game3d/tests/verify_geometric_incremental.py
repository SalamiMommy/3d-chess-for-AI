
import sys
import os
import numpy as np
import logging

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, SIZE
from game3d.game.turnmove import make_move
from game3d.common.coord_utils import coords_to_keys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_geometric_invalidation():
    logger.info("--- Testing Geometric Invalidation ---")
    
    board = Board()
    state = GameState(board, Color.WHITE)
    
    # Clear board
    # We can't easily clear, so let's just make a custom setup
    # Manually managing cache is risky, so we rely on standard moves or clean init.
    
    # Let's use internal methods to set up a scenario
    occ_cache = state.cache_manager.occupancy_cache
    
    # 1. Setup:
    # White Rook at [0, 0, 0]
    # White Bishop at [2, 0, 0] (Not aligned with interaction at [0,5,0])
    # Black Target at [0, 5, 0]
    
    # Clear pieces hack (unsafe but necessary for isolated test)
    # occ_cache.clear() # No clear method exposed typically, let's just ignore existing
    
    # Better: Use set_position to place pieces
    # We should ensure we don't overwrite King if needed, but for raw generation it's fine.
    
    rook_pos = np.array([0, 0, 0], dtype=COORD_DTYPE)
    bishop_pos = np.array([2, 0, 0], dtype=COORD_DTYPE)
    target_pos = np.array([0, 5, 0], dtype=COORD_DTYPE)
    mover_pos = np.array([0, 4, 0], dtype=COORD_DTYPE)
    
    # Set pieces
    # set_position takes (coord, piece_data) where piece_data is [type, color]
    from game3d.common.shared_types import PIECE_TYPE_DTYPE
    
    rook_data = np.array([PieceType.ROOK, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
    bishop_data = np.array([PieceType.BISHOP, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
    mover_data = np.array([PieceType.PAWN, Color.BLACK], dtype=PIECE_TYPE_DTYPE)
    target_data = np.array([PieceType.PAWN, Color.BLACK], dtype=PIECE_TYPE_DTYPE)
    
    white_king_data = np.array([PieceType.KING, Color.WHITE], dtype=PIECE_TYPE_DTYPE)
    black_king_data = np.array([PieceType.KING, Color.BLACK], dtype=PIECE_TYPE_DTYPE)

    occ_cache.set_position(rook_pos, rook_data)
    occ_cache.set_position(bishop_pos, bishop_data)
    occ_cache.set_position(mover_pos, mover_data) # Mover
    occ_cache.set_position(target_pos, target_data) # Target
    
    # Ensure Kings exist for safety
    occ_cache.set_position(np.array([8,8,8], dtype=COORD_DTYPE), white_king_data)
    occ_cache.set_position(np.array([8,8,0], dtype=COORD_DTYPE), black_king_data)
    
    state.cache_manager._move_counter = 0 # specific test state
    
    # 2. Generate Moves (Populates Cache)
    # This runs generate_fused -> incremental -> populates cache
    moves = state.gen_moves()
    logger.info(f"Initial moves generated: {len(moves)}")
    
    # Verify cache is populated
    rook_key = int(coords_to_keys(rook_pos.reshape(1,3))[0])
    bishop_key = int(coords_to_keys(bishop_pos.reshape(1,3))[0])
    
    c_idx_white = 0
    rook_id = (c_idx_white, rook_key)
    bishop_id = (c_idx_white, bishop_key)
    
    move_cache = state.cache_manager.move_cache
    
    if rook_id not in move_cache._piece_moves_cache:
        logger.error("Rook moves not cached!")
        return
        
    if bishop_id not in move_cache._piece_moves_cache:
        logger.error("Bishop moves not cached!")
        return
        
    logger.info("Cache populated correctly.")
    
    # 3. Apply Move
    # Black Pawn at [0,4,0] moves to [0,5,0] (Capture)
    # This interaction is at [0,4,0] and [0,5,0].
    # Rook at [0,0,0] IS ALIGNED (x=0).
    # Bishop at [2,0,0] IS NOT ALIGNED with [0,4,0] or [0,5,0].
    # (Checking: [2,0,0] vs [0,4,0]. dx=2, dy=4, dz=0. Not Ortho. Not Diag (2!=4). Not Triag.)
    
    move_arr = np.array([0, 4, 0, 0, 5, 0], dtype=COORD_DTYPE)
    
    # Manually trigger invalidation (easier than full move correctness with king check etc)
    # Or just call manager.apply_move_incremental simulation
    
    # Let's perform the move mechanics logic relevant to invalidation
    # We'll use manager._invalidate_affected_piece_moves directly to test logic
    affected_coords = np.array([[0, 4, 0], [0, 5, 0]], dtype=COORD_DTYPE)
    
    # BEFORE: Clear previous affected list
    move_cache._affected_coord_keys_list = []
    
    state.cache_manager._invalidate_affected_piece_moves(affected_coords, Color.BLACK, state) # updates both colors
    
    # 4. Check Updates
    affected_list = move_cache.get_affected_pieces(Color.WHITE) # Returns KEYS
    affected_set = set(affected_list.tolist())
    
    logger.info(f"Affected Keys: {affected_set}")
    logger.info(f"Rook Key: {rook_key}, Bishop Key: {bishop_key}")
    
    is_rook_invalid = rook_key in affected_set
    is_bishop_invalid = bishop_key in affected_set
    
    if is_rook_invalid:
        logger.info("SUCCESS: Rook correctly invalidated (Aligned)")
    else:
        logger.error("FAILURE: Rook NOT invalidated!")
        
    if not is_bishop_invalid:
        logger.info("SUCCESS: Bishop correctly preserved (Not Aligned)")
    else:
        logger.error("FAILURE: Bishop improperly invalidated!")

def verify_incremental_correctness():
    logger.info("\n--- Testing Incremental Gen Correctness ---")
    board = Board()
    state = GameState(board, Color.WHITE)
    
    # Generate initial moves
    moves_1 = state.gen_moves()
    
    # Move a white pawn
    # Find a valid move
    mv = moves_1[0]
    make_move(state, mv)
    
    # Now Black to move. 
    moves_2 = state.gen_moves()
    # This relied on incremental update of Black's moves from previous turn?
    # No, cache is by color. Black moves generated from scratch? 
    # Or from previous cache if it existed.
    # Initially cache is empty.
    
    # Let's play a few plies
    move_count = 0
    for _ in range(5):
        mvs = state.gen_moves()
        if len(mvs) == 0: break
        mv = mvs[0]
        make_move(state, mv)
        move_count += 1
        
    logger.info(f"Played {move_count} moves. Cache should be primed.")
    
    # Compare Incremental output vs fresh generation
    # 1. Get current incremental output
    inc_moves = state.gen_moves()
    # Sort for comparison
    inc_moves = inc_moves[np.lexsort((inc_moves[:,5], inc_moves[:,4], inc_moves[:,3], inc_moves[:,2], inc_moves[:,1], inc_moves[:,0]))]
    
    # 2. Clear cache and regen
    state.cache_manager.move_cache.invalidate()
    full_moves = state.gen_moves()
    full_moves = full_moves[np.lexsort((full_moves[:,5], full_moves[:,4], full_moves[:,3], full_moves[:,2], full_moves[:,1], full_moves[:,0]))]
    
    if inc_moves.shape == full_moves.shape and np.all(inc_moves == full_moves):
        logger.info("SUCCESS: Incremental Generation matches Full Generation")
    else:
        logger.error("FAILURE: Mismatch!")
        logger.error(f"Inc: {inc_moves.shape}, Full: {full_moves.shape}")
        
        # Identify differences
        # Using set logic on rows
        def to_set(arr):
             return set([tuple(x) for x in arr])
             
        inc_set = to_set(inc_moves)
        full_set = to_set(full_moves)
        
        missing = full_set - inc_set
        extra = inc_set - full_set
        
        logger.error(f"Missing in Inc ({len(missing)}): {list(missing)[:5]}")
        logger.error(f"Extra in Inc ({len(extra)}): {list(extra)[:5]}")

if __name__ == "__main__":
    verify_geometric_invalidation()
    verify_incremental_correctness()
