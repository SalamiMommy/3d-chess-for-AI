
import numpy as np
import logging
from game3d.board.board import Board
from game3d.cache.manager import get_cache_manager
from game3d.common.shared_types import PieceType, Color
from game3d.movement.generator import generate_legal_moves_for_piece

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reproduce():
    # Initialize board
    board = Board.startpos()
    manager = get_cache_manager(board)
    
    # Check piece count
    coords = manager.occupancy_cache.get_positions(Color.WHITE)
    count = len(coords)
    logger.info(f"White pieces: {count}")
    
    coords_black = manager.occupancy_cache.get_positions(Color.BLACK)
    count_black = len(coords_black)
    logger.info(f"Black pieces: {count_black}")
    
    total = count + count_black
    logger.info(f"Total pieces: {total}")
    
    if total == 486:
        logger.info("Confirmed: 486 pieces found (expected 462).")
    else:
        logger.info(f"Unexpected piece count: {total}")

    # Check wall moves
    # Find a wall piece
    # Rank 2 (z=1 for white)
    # Top-left block is at x=1, y=1 (anchor)
    # Non-anchor parts: (1,2), (2,1), (2,2)
    
    anchor = np.array([1, 1, 1])
    part = np.array([1, 2, 1])
    
    # Check if they are walls
    type_anchor = manager.occupancy_cache.get_type_at(*anchor)
    type_part = manager.occupancy_cache.get_type_at(*part)
    
    logger.info(f"Piece at {anchor}: {type_anchor} (Expected {PieceType.WALL})")
    logger.info(f"Piece at {part}: {type_part} (Expected {PieceType.WALL})")
    
    if type_anchor == PieceType.WALL and type_part == PieceType.WALL:
        # Generate moves for anchor
        moves_anchor = generate_legal_moves_for_piece(manager, anchor)
        logger.info(f"Moves for anchor {anchor}: {len(moves_anchor)}")
        
        # Generate moves for part
        moves_part = generate_legal_moves_for_piece(manager, part)
        logger.info(f"Moves for part {part}: {len(moves_part)}")
        
        if len(moves_part) > 0:
            logger.info("ISSUE REPRODUCED: Non-anchor wall part generated moves!")
        else:
            logger.info("Non-anchor wall part generated 0 moves (Good).")
            
if __name__ == "__main__":
    reproduce()
