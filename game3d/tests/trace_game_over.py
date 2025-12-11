"""
Direct investigation: Why does the game continue after "No legal moves" is detected?

Key hypothesis: The "No legal moves for BLACK" log is generated during move regeneration
INSIDE make_move(), but is_game_over() is not checked until the NEXT loop iteration.
"""
import logging
import numpy as np

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs but keep movecache
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('game3d.movement').setLevel(logging.WARNING)
logging.getLogger('game3d.cache.manager').setLevel(logging.WARNING)
logging.getLogger('game3d.pieces').setLevel(logging.WARNING)
logging.getLogger('game3d.core').setLevel(logging.WARNING)
# Keep movecache at DEBUG to see "No legal moves" message
logging.getLogger('game3d.cache.caches.movecache').setLevel(logging.DEBUG)
logging.getLogger('game3d.game.terminal').setLevel(logging.DEBUG)


def trace_is_game_over_and_legal_moves():
    """
    Add instrumentation to understand when is_game_over is called vs when
    legal moves are generated.
    """
    from game3d.game.factory import start_game_state
    from game3d.main_game import OptimizedGame3D
    from game3d.common.shared_types import Color, COORD_DTYPE
    from game3d.movement.movepiece import Move
    from game3d.game import terminal
    
    # Patch is_game_over to add logging
    original_is_game_over = terminal.is_game_over
    call_count = [0]
    
    def patched_is_game_over(game_state):
        call_count[0] += 1
        occ = game_state.cache_manager.occupancy_cache
        coords, _, _ = occ.get_all_occupied_vectorized()
        piece_count = len(coords)
        
        result = original_is_game_over(game_state)
        logger.info(f"[TRACE] is_game_over() call #{call_count[0]}: color={Color(game_state.color).name}, pieces={piece_count}, result={result}")
        return result
    
    terminal.is_game_over = patched_is_game_over
    
    # Also patch the main_game import
    from game3d import main_game
    main_game._terminal_is_game_over = patched_is_game_over
    
    try:
        initial_state = start_game_state()
        game = OptimizedGame3D(board=initial_state.board, cache=initial_state.cache_manager)
        
        move_count = 0
        max_moves = 1000  # Run longer to potentially hit checkmate
        
        logger.info("=" * 70)
        logger.info("Starting game loop - watching for 'No legal moves' log")
        logger.info("=" * 70)
        
        last_piece_count = 486
        
        while move_count < max_moves:
            # Get piece count BEFORE is_game_over check
            occ = game.state.cache_manager.occupancy_cache
            coords, _, _ = occ.get_all_occupied_vectorized()
            piece_count = len(coords)
            
            if piece_count != last_piece_count:
                logger.info(f"[PIECE_COUNT] Changed: {last_piece_count} -> {piece_count}")
                last_piece_count = piece_count
            
            # Check game over
            if game.is_game_over():
                logger.info(f"[GAME_OVER] Loop detected game over at move {move_count}, pieces={piece_count}")
                break
            
            # Get legal moves
            legal_moves = game.state.legal_moves
            if legal_moves.size == 0:
                logger.warning(f"[BUG?] legal_moves.size == 0 at move {move_count} but is_game_over was False!")
                logger.warning(f"  Color: {Color(game.state.color).name}, Pieces: {piece_count}")
                break
            
            # Filter by color (as parallel_self_play does)
            from_coords = legal_moves[:, :3].astype(COORD_DTYPE)
            from_colors, _ = occ.batch_get_attributes(from_coords)
            valid_mask = from_colors == game.state.color
            
            if not np.any(valid_mask):
                logger.warning(f"[BUG?] No valid moves after color filter at move {move_count}")
                logger.warning(f"  legal_moves: {len(legal_moves)}, valid after filter: 0")
                break
            
            valid_moves = legal_moves[valid_mask]
            
            # Pick first move
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
            
            game._state = receipt.new_state
            game._state._legal_moves_cache = None
            move_count += 1
            
            # Progress indicator
            if move_count % 100 == 0:
                logger.info(f"[PROGRESS] Move {move_count}, pieces={piece_count}, color={Color(game.state.color).name}")
        
        logger.info(f"\n[RESULT] Loop exited after {move_count} moves")
        logger.info(f"[RESULT] is_game_over() was called {call_count[0]} times")
        
    finally:
        # Restore original function
        terminal.is_game_over = original_is_game_over


def analyze_receipts():
    """
    Check if submit_move receipt correctly reports is_game_over.
    """
    from game3d.game.factory import start_game_state
    from game3d.main_game import OptimizedGame3D
    from game3d.common.shared_types import Color, COORD_DTYPE
    from game3d.movement.movepiece import Move
    
    initial_state = start_game_state()
    game = OptimizedGame3D(board=initial_state.board, cache=initial_state.cache_manager)
    
    logger.info("=" * 70)
    logger.info("Analyzing submit_move receipt.is_game_over")
    logger.info("=" * 70)
    
    move_count = 0
    max_moves = 100
    
    while move_count < max_moves:
        if game.is_game_over():
            break
        
        legal_moves = game.state.legal_moves
        if legal_moves.size == 0:
            break
        
        # Filter and pick move
        from_coords = legal_moves[:, :3].astype(COORD_DTYPE)
        occ = game.state.cache_manager.occupancy_cache
        from_colors, _ = occ.batch_get_attributes(from_coords)
        valid_mask = from_colors == game.state.color
        
        if not np.any(valid_mask):
            break
        
        valid_moves = legal_moves[valid_mask]
        chosen_move = valid_moves[0]
        submit_move = Move(
            chosen_move[:3].astype(COORD_DTYPE),
            chosen_move[3:6].astype(COORD_DTYPE)
        )
        
        try:
            receipt = game.submit_move(submit_move)
            
            # Critical check: Does receipt say game is over?
            if receipt.is_game_over:
                logger.info(f"[RECEIPT] Move {move_count}: receipt.is_game_over = True")
                logger.info(f"  But we're about to continue the loop!")
                logger.info(f"  receipt.result = {receipt.result}")
            
        except Exception as e:
            logger.error(f"Move failed: {e}")
            break
        
        game._state = receipt.new_state
        game._state._legal_moves_cache = None
        move_count += 1
    
    logger.info(f"\n[RESULT] Analyzed {move_count} moves")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "trace":
            trace_is_game_over_and_legal_moves()
        elif test == "receipts":
            analyze_receipts()
        else:
            print(f"Unknown test: {test}")
    else:
        trace_is_game_over_and_legal_moves()
