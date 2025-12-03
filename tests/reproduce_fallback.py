
import numpy as np
import logging
import time
from game3d.board.board import Board
from game3d.cache.manager import get_cache_manager
from game3d.common.shared_types import PieceType, Color
from game3d.attacks import check
from game3d.game.gamestate import GameState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkeypatch _square_attacked_by_slow to track calls
original_slow = check._square_attacked_by_slow
slow_call_count = 0

def mocked_slow(board, square, attacker_color, cache=None):
    global slow_call_count
    slow_call_count += 1
    # logger.warning(f"Fallback to slow check! Square: {square}, Attacker: {attacker_color}")
    return original_slow(board, square, attacker_color, cache)

check._square_attacked_by_slow = mocked_slow

def reproduce():
    global slow_call_count
    
    # Initialize board
    logger.info("Initializing board...")
    board = Board.startpos()
    manager = get_cache_manager(board)
    game_state = GameState(board, Color.WHITE, manager)
    
    # Initial check
    logger.info("Initial check status...")
    check.get_check_status(board, Color.WHITE, Color.WHITE, manager)
    logger.info(f"Slow calls after initial check: {slow_call_count}")
    
    # Reset counter
    slow_call_count = 0
    
    # Make a move (White Pawn push)
    # Find a pawn
    white_pawns = manager.occupancy_cache.get_positions(Color.WHITE)
    pawn_pos = None
    for pos in white_pawns:
        if manager.occupancy_cache.get_type_at(*pos) == PieceType.PAWN:
            pawn_pos = pos
            break
            
    if pawn_pos is None:
        logger.error("No white pawns found!")
        return

    # REMOVE PRIESTS to force check detection
    logger.info("Removing all white priests...")
    priest_positions = []
    for pos in manager.occupancy_cache.get_positions(Color.WHITE):
        if manager.occupancy_cache.get_type_at(*pos) == PieceType.PRIEST:
            priest_positions.append(pos)
    
    for pos in priest_positions:
        manager.occupancy_cache.set_position(pos, None)
        
    logger.info(f"Removed {len(priest_positions)} priests.")
    
    # Verify no priests
    if manager.occupancy_cache.has_priest(Color.WHITE):
        logger.error("Failed to remove all priests!")
        return

    logger.info(f"Moving pawn at {pawn_pos}")
    
    # Simulate a move check (e.g. is this move safe?)
    # Move forward 1 step
    to_pos = pawn_pos.copy()
    to_pos[1] += 1 # Move in Y
    
    move = np.concatenate([pawn_pos, to_pos])
    
    logger.info("Generating legal moves (triggers filter_safe_moves_optimized)...")
    from game3d.movement.generator import generate_legal_moves
    
    # This should trigger our fix in filter_safe_moves_optimized
    moves = generate_legal_moves(game_state)
    
    logger.info(f"Generated {len(moves)} moves.")
    logger.info(f"Slow calls during generation: {slow_call_count}")
    
    # Check cache stats
    stats = manager.move_cache.get_statistics()
    logger.info(f"Cache Stats: {stats}")
    
    if slow_call_count > 0:
        logger.info("ISSUE REPRODUCED: Fallback to slow check occurred.")
        
        # Investigate why
        # Check if cache is dirty
        affected_white = manager.move_cache.get_affected_pieces(Color.WHITE)
        affected_black = manager.move_cache.get_affected_pieces(Color.BLACK)
        logger.info(f"Affected White Pieces: {len(affected_white)}")
        logger.info(f"Affected Black Pieces: {len(affected_black)}")
        
        if len(affected_black) > 0:
             logger.info("Black cache is dirty! This explains why get_pseudolegal_moves(BLACK) returns None.")
             
    else:
        logger.info("No fallback occurred (Good).")

if __name__ == "__main__":
    reproduce()
