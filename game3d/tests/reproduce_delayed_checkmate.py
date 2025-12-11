"""
Reproduction script for delayed checkmate bug.

Issue: When Black has no legal moves at 110 pieces, the game should end immediately.
Instead, it continues for 2 more moves until 108 pieces.

This script traces:
1. When is_game_over() returns True vs False
2. When "No legal moves" is logged
3. What color is active when these checks happen
"""
import logging
import numpy as np
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('numba').setLevel(logging.WARNING)

def setup_checkmate_position():
    """
    Set up a position where Black is in checkmate.
    - Black King at (4,4,8) with no escape squares
    - White Rook checking along a file
    - All Black's moves are blocked by king safety
    """
    from game3d.game.factory import start_game_state
    from game3d.main_game import OptimizedGame3D
    from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
    
    gs = start_game_state()
    game = OptimizedGame3D(board=gs.board, cache=gs.cache_manager)
    
    # Clear the board
    occ_cache = game.state.cache_manager.occupancy_cache
    occ_cache.clear()
    game.state.cache_manager.move_cache.invalidate()
    
    # Set up minimal checkmate position
    # Black King at center top
    bk_pos = np.array([4, 4, 8], dtype=COORD_DTYPE)
    occ_cache.set_position(bk_pos, np.array([PieceType.KING.value, Color.BLACK.value]))
    
    # White King far away
    wk_pos = np.array([4, 4, 0], dtype=COORD_DTYPE)
    occ_cache.set_position(wk_pos, np.array([PieceType.KING.value, Color.WHITE.value]))
    
    # White Rooks giving checkmate
    wr1_pos = np.array([3, 4, 8], dtype=COORD_DTYPE)  # Checking along y
    occ_cache.set_position(wr1_pos, np.array([PieceType.ROOK.value, Color.WHITE.value]))
    
    wr2_pos = np.array([5, 4, 8], dtype=COORD_DTYPE)  # Cover escape
    occ_cache.set_position(wr2_pos, np.array([PieceType.ROOK.value, Color.WHITE.value]))
    
    # Invalidate caches
    game.state.cache_manager.move_cache.invalidate()
    
    return game


def trace_game_loop():
    """Simulate the game loop and trace when is_game_over is checked."""
    from game3d.game.factory import start_game_state
    from game3d.main_game import OptimizedGame3D
    from game3d.common.shared_types import Color
    from game3d.game.terminal import is_game_over, is_check
    
    gs = start_game_state()
    game = OptimizedGame3D(board=gs.board, cache=gs.cache_manager)
    
    logger.info("=" * 60)
    logger.info("Starting game loop trace")
    logger.info("=" * 60)
    
    move_count = 0
    max_moves = 20  # Limit for testing
    
    while move_count < max_moves:
        state = game.state
        current_color = state.color
        
        # Count pieces
        occ_cache = state.cache_manager.occupancy_cache
        coords, _, colors = occ_cache.get_all_occupied_vectorized()
        piece_count = len(coords)
        
        logger.info(f"\n--- Loop iteration {move_count} ---")
        logger.info(f"Current color: {Color(current_color).name}")
        logger.info(f"Piece count: {piece_count}")
        
        # Check is_game_over BEFORE accessing legal_moves
        logger.info("Checking is_game_over()...")
        game_over = game.is_game_over()
        logger.info(f"is_game_over() = {game_over}")
        
        if game_over:
            logger.info("GAME OVER - exiting loop")
            break
        
        # Get legal moves
        logger.info("Getting legal_moves...")
        legal_moves = state.legal_moves
        logger.info(f"legal_moves.size = {legal_moves.size}")
        
        if legal_moves.size == 0:
            logger.info("No legal moves - but is_game_over() was False!")
            logger.info(f"is_check() = {is_check(state)}")
            break
        
        # Make a random move to progress the game
        chosen_idx = np.random.randint(0, len(legal_moves))
        chosen_move = legal_moves[chosen_idx]
        
        from game3d.movement.movepiece import Move
        move = Move(chosen_move[:3], chosen_move[3:6])
        
        logger.info(f"Making move: {move.from_coord} -> {move.to_coord}")
        
        try:
            receipt = game.submit_move(move)
            move_count += 1
        except Exception as e:
            logger.error(f"Move failed: {e}")
            break
    
    logger.info(f"\nGame ended after {move_count} moves")
    return move_count


def trace_state_color_flow():
    """
    Trace the color changes during move execution to see if
    is_game_over checks the right color.
    """
    from game3d.game.factory import start_game_state
    from game3d.main_game import OptimizedGame3D
    from game3d.common.shared_types import Color
    from game3d.game.terminal import is_game_over, is_check
    
    gs = start_game_state()
    game = OptimizedGame3D(board=gs.board, cache=gs.cache_manager)
    
    logger.info("=" * 60)
    logger.info("Tracing state.color flow during execution")
    logger.info("=" * 60)
    
    state = game.state
    logger.info(f"Initial color: {Color(state.color).name}")
    
    # Get first legal move for White
    legal_moves = state.legal_moves
    if legal_moves.size == 0:
        logger.error("No moves at start!")
        return
    
    move = legal_moves[0]
    
    from game3d.movement.movepiece import Move
    mv = Move(move[:3], move[3:6])
    
    logger.info(f"Before submit_move: state.color = {Color(state.color).name}")
    
    receipt = game.submit_move(mv)
    
    logger.info(f"After submit_move: game.state.color = {Color(game.state.color).name}")
    logger.info(f"receipt.new_state.color = {Color(receipt.new_state.color).name}")
    
    # Now check is_game_over on the new state
    logger.info(f"\nChecking is_game_over on game.state...")
    logger.info(f"game.state.color = {Color(game.state.color).name}")
    result = game.is_game_over()
    logger.info(f"is_game_over() = {result}")
    
    # Check legal_moves
    logger.info(f"\nChecking legal_moves on game.state...")
    moves = game.state.legal_moves
    logger.info(f"legal_moves.size = {moves.size}")


def check_parallel_self_play_flow():
    """
    Reproduce the exact flow from parallel_self_play.py to find the bug.
    """
    from game3d.game.factory import start_game_state
    from game3d.main_game import OptimizedGame3D
    from game3d.common.shared_types import Color, COORD_DTYPE
    from game3d.movement.movepiece import Move
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("Reproducing parallel_self_play.py game loop flow")
    logger.info("=" * 60)
    
    initial_state = start_game_state()
    game = OptimizedGame3D(board=initial_state.board, cache=initial_state.cache_manager)
    
    move_count = 0
    max_moves = 50
    
    # Exact flow from parallel_self_play.py lines 221-327
    while move_count < max_moves and not game.is_game_over():
        
        legal_moves = game.state.legal_moves
        if legal_moves.size == 0:
            logger.warning(f"legal_moves.size == 0 at move {move_count}, but is_game_over was False!")
            logger.warning(f"Current color: {Color(game.state.color).name}")
            break
        
        # Exact flow from lines 242-252
        from_coords = legal_moves[:, :3].astype(COORD_DTYPE)
        to_coords = legal_moves[:, 3:6].astype(COORD_DTYPE)
        
        occ_cache = game.state.cache_manager.occupancy_cache
        from_colors, _ = occ_cache.batch_get_attributes(from_coords)
        
        # Filter valid moves - THIS IS THE KEY LINE!
        valid_mask = from_colors == game.state.color
        if not np.any(valid_mask):
            logger.warning(f"No valid moves after color filter at move {move_count}")
            logger.warning(f"legal_moves had {len(legal_moves)} moves but none match current color")
            logger.warning(f"Current color: {Color(game.state.color).name}")
            logger.warning(f"from_colors: {set(from_colors.tolist())}")
            break
        
        valid_moves = legal_moves[valid_mask]
        
        # Select move (simplified - just pick first)
        chosen_move = valid_moves[0]
        submit_move = Move(
            chosen_move[:3].astype(COORD_DTYPE),
            chosen_move[3:6].astype(COORD_DTYPE)
        )
        
        try:
            receipt = game.submit_move(submit_move)
        except Exception as e:
            logger.error(f"Move failed: {e}")
            break
        
        # Exact flow from line 325
        game._state = receipt.new_state
        game._state._legal_moves_cache = None
        move_count += 1
        
        if move_count % 10 == 0:
            occ_cache = game.state.cache_manager.occupancy_cache
            coords, _, _ = occ_cache.get_all_occupied_vectorized()
            logger.info(f"Move {move_count}: {len(coords)} pieces, color: {Color(game.state.color).name}")
    
    logger.info(f"\nLoop exited after {move_count} moves")
    logger.info(f"Final is_game_over: {game.is_game_over()}")
    logger.info(f"Final color: {Color(game.state.color).name}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "loop":
            trace_game_loop()
        elif test == "color":
            trace_state_color_flow()
        elif test == "parallel":
            check_parallel_self_play_flow()
        else:
            print(f"Unknown test: {test}")
            print("Options: loop, color, parallel")
    else:
        # Run all tests
        print("\n=== Test 1: Game Loop Trace ===\n")
        trace_game_loop()
        
        print("\n\n=== Test 2: Color Flow ===\n")
        trace_state_color_flow()
        
        print("\n\n=== Test 3: Parallel Self-Play Flow ===\n")
        check_parallel_self_play_flow()
