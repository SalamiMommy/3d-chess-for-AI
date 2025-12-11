"""
Debug script to trace the exact sequence when checkmate occurs.
"""
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress everything except our traces
logging.getLogger('numba').setLevel(logging.ERROR)
for name in ['game3d.movement', 'game3d.cache.manager', 'game3d.pieces', 
             'game3d.core', 'game3d.attacks', 'game3d.board']:
    logging.getLogger(name).setLevel(logging.WARNING)

# Keep these at DEBUG
logging.getLogger('game3d.cache.caches.movecache').setLevel(logging.DEBUG)
logging.getLogger('game3d.game.terminal').setLevel(logging.DEBUG)


def trace_checkmate_scenario():
    """
    Set up a position close to checkmate and trace exactly what happens.
    """
    from game3d.game.gamestate import GameState
    from game3d.board.board import Board
    from game3d.main_game import OptimizedGame3D
    from game3d.cache.manager import OptimizedCacheManager
    from game3d.common.shared_types import Color, PieceType, COORD_DTYPE
    from game3d.movement.movepiece import Move
    
    # Create an empty board
    board = Board.empty()
    cache = OptimizedCacheManager(board, Color.WHITE)
    game = OptimizedGame3D(board=board, cache=cache)
    
    occ_cache = game.state.cache_manager.occupancy_cache
    
    # Set up a checkmate position where Black is in checkmate
    # Black King at (4, 4, 8) - trapped in corner
    # White King at (4, 4, 0) - safe far away
    # White Rooks on 3rd and 5th file, 8th rank
    
    logger.info("Setting up checkmate position...")
    
    # Black King - will be in checkmate
    bk = np.array([4, 4, 8], dtype=COORD_DTYPE)
    occ_cache.set_position(bk, np.array([PieceType.KING.value, Color.BLACK.value]))
    
    # White King - safe
    wk = np.array([4, 4, 0], dtype=COORD_DTYPE)
    occ_cache.set_position(wk, np.array([PieceType.KING.value, Color.WHITE.value]))
    
    # White Rooks giving check and covering escape
    wr1 = np.array([4, 0, 8], dtype=COORD_DTYPE)  # On rank 8, file 0
    occ_cache.set_position(wr1, np.array([PieceType.ROOK.value, Color.WHITE.value]))
    
    wr2 = np.array([4, 8, 7], dtype=COORD_DTYPE)  # Covering escape
    occ_cache.set_position(wr2, np.array([PieceType.ROOK.value, Color.WHITE.value]))
    
    # Another White Rook for more coverage
    wr3 = np.array([3, 4, 7], dtype=COORD_DTYPE)
    occ_cache.set_position(wr3, np.array([PieceType.ROOK.value, Color.WHITE.value]))
    
    # Invalidate caches
    game.state.cache_manager.move_cache.invalidate()
    
    # Set turn to Black
    game._state.color = Color.BLACK
    game._state.turn_number = 100
    
    logger.info("=" * 70)
    logger.info(f"Position set. Current turn: {Color(game.state.color).name}")
    logger.info("=" * 70)
    
    # Now check is_game_over - this should be True if Black is in checkmate
    logger.info("\nCalling is_game_over()...")
    result = game.is_game_over()
    logger.info(f"is_game_over() returned: {result}")
    
    # Get legal moves
    logger.info("\nGetting legal_moves...")
    moves = game.state.legal_moves
    logger.info(f"legal_moves.size: {moves.size}")
    
    # Check if in check
    from game3d.game.terminal import is_check
    logger.info(f"is_check(): {is_check(game.state)}")


def trace_piece_count_changes():
    """
    Run game and trace when piece count changes to understand flow.
    """
    from game3d.game.factory import start_game_state
    from game3d.main_game import OptimizedGame3D
    from game3d.common.shared_types import Color, COORD_DTYPE
    from game3d.movement.movepiece import Move
    from game3d.game import terminal
    
    # Add trace to store_legal_moves
    from game3d.cache.caches import movecache
    original_store = movecache.MoveCache.store_legal_moves
    
    def traced_store(self, color, moves):
        piece_count = len(self.cache_manager.occupancy_cache.get_positions(color))
        if moves.size == 0:
            logger.warning(f"[STORE_TRACE] store_legal_moves: {Color(color).name} has 0 moves, {piece_count} pieces")
        return original_store(self, color, moves)
    
    movecache.MoveCache.store_legal_moves = traced_store
    
    try:
        gs = start_game_state()
        game = OptimizedGame3D(board=gs.board, cache=gs.cache_manager)
        
        move_count = 0
        max_moves = 2000
        
        logger.info("=" * 70)
        logger.info("Starting game - watching for zero moves")
        logger.info("=" * 70)
        
        while move_count < max_moves and not game.is_game_over():
            
            moves = game.state.legal_moves
            if moves.size == 0:
                logger.warning(f"[LOOP] legal_moves.size == 0 at move {move_count}")
                logger.warning(f"  But is_game_over() returned False!")
                break
            
            # Get piece count
            occ = game.state.cache_manager.occupancy_cache
            coords, _, _ = occ.get_all_occupied_vectorized()
            
            # Filter valid moves
            from_coords = moves[:, :3].astype(COORD_DTYPE)
            from_colors, _ = occ.batch_get_attributes(from_coords)
            valid_mask = from_colors == game.state.color
            
            if not np.any(valid_mask):
                logger.warning(f"[LOOP] No valid moves after filter at move {move_count}")
                break
            
            valid_moves = moves[valid_mask]
            chosen_move = valid_moves[0]
            submit_move = Move(chosen_move[:3], chosen_move[3:6])
            
            try:
                receipt = game.submit_move(submit_move)
                
                # Check receipt BEFORE updating state
                if receipt.is_game_over:
                    logger.info(f"[RECEIPT] Game over detected at move {move_count}")
                    logger.info(f"  result={receipt.result}")
                    break
                    
            except Exception as e:
                logger.error(f"Move failed: {e}")
                break
            
            game._state = receipt.new_state
            move_count += 1
            
            if move_count % 100 == 0:
                logger.info(f"[PROGRESS] Move {move_count}, {len(coords)} pieces")
        
        logger.info(f"\n[RESULT] Game ended after {move_count} moves")
        
    finally:
        movecache.MoveCache.store_legal_moves = original_store


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "checkmate":
        trace_checkmate_scenario()
    else:
        trace_piece_count_changes()
