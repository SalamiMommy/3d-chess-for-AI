
import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append('/home/salamimommy/Documents/code/3d-chess-for-AI')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from game3d.common.shared_types import PieceType, COORD_DTYPE, BOOL_DTYPE, INDEX_DTYPE, N_PIECE_TYPES
from training.opponents import ATTACK_VECTORS, VECTOR_COUNTS, IS_SLIDER, _compute_check_potential_vectorized

def test_initialization():
    logger.info("Testing initialization of attack tables...")
    
    # Check standard pieces
    assert VECTOR_COUNTS[0, PieceType.KNIGHT.value] == 24, "Knight should have 24 vectors"
    assert VECTOR_COUNTS[0, PieceType.BISHOP.value] == 12, "Bishop should have 12 vectors"
    assert VECTOR_COUNTS[0, PieceType.ROOK.value] == 6, "Rook should have 6 vectors"
    assert VECTOR_COUNTS[0, PieceType.QUEEN.value] == 18, "Queen should have 18 vectors"
    import game3d.pieces.pieces as all_pieces
    if hasattr(all_pieces, 'KING_MOVEMENT_VECTORS'):
        print(f"all_pieces.KING_MOVEMENT_VECTORS shape: {all_pieces.KING_MOVEMENT_VECTORS.shape}")
        print(f"all_pieces.KING_MOVEMENT_VECTORS: {all_pieces.KING_MOVEMENT_VECTORS}")
    else:
        print("all_pieces.KING_MOVEMENT_VECTORS not found")
        
    print(f"King vectors: {VECTOR_COUNTS[0, PieceType.KING.value]}")
    assert VECTOR_COUNTS[0, PieceType.KING.value] == 26, f"King should have 26 vectors, got {VECTOR_COUNTS[0, PieceType.KING.value]}"
    
    # Check Pawn (color specific)
    assert VECTOR_COUNTS[0, PieceType.PAWN.value] == 4, "White Pawn should have 4 attack vectors"
    assert VECTOR_COUNTS[1, PieceType.PAWN.value] == 4, "Black Pawn should have 4 attack vectors"
    
    # Check Slider flags
    assert IS_SLIDER[PieceType.BISHOP.value], "Bishop should be slider"
    assert IS_SLIDER[PieceType.ROOK.value], "Rook should be slider"
    assert not IS_SLIDER[PieceType.KNIGHT.value], "Knight should not be slider"
    
    logger.info("Initialization checks passed!")

def test_check_potential():
    logger.info("Testing _compute_check_potential_vectorized...")
    
    # Setup mock data
    n_moves = 5
    to_coords = np.zeros((n_moves, 3), dtype=COORD_DTYPE)
    piece_types = np.zeros(n_moves, dtype=np.int8)
    
    # 1. Knight check
    # King at (4,4,4)
    enemy_king_pos = np.array([4, 4, 4], dtype=COORD_DTYPE)
    
    # Knight at (2,3,4) -> (4-2)^2 + (4-3)^2 + (4-4)^2 = 4 + 1 + 0 = 5 (Knight jump)
    to_coords[0] = [2, 3, 4]
    piece_types[0] = PieceType.KNIGHT.value
    
    # 2. Rook check (unblocked)
    # Rook at (4, 0, 4) -> (4,4,4) is straight line
    to_coords[1] = [4, 0, 4]
    piece_types[1] = PieceType.ROOK.value
    
    # 3. Rook check (blocked)
    # Rook at (0, 4, 4) -> (4,4,4) is straight line
    to_coords[2] = [0, 4, 4]
    piece_types[2] = PieceType.ROOK.value
    
    # 4. Pawn check (White attacking Black King)
    # Pawn at (3, 3, 3) -> King at (4,4,4)
    # Diff = (1, 1, 1). White pawn attacks (1,1,1)
    to_coords[3] = [3, 3, 3]
    piece_types[3] = PieceType.PAWN.value
    
    # 5. Non-check
    to_coords[4] = [0, 0, 0]
    piece_types[4] = PieceType.KNIGHT.value
    
    # Occupancy grid
    occupancy_grid = np.zeros((9, 9, 9), dtype=np.uint8)
    # Block the 3rd move (Rook at 0,4,4 attacking 4,4,4)
    # Path: 1,4,4; 2,4,4; 3,4,4. Block at 2,4,4
    occupancy_grid[2, 4, 4] = 1
    
    # Run computation
    rewards = _compute_check_potential_vectorized(
        to_coords, piece_types, enemy_king_pos, 1.0,
        occupancy_grid, ATTACK_VECTORS, VECTOR_COUNTS, IS_SLIDER,
        0 # Attacker is White
    )
    
    print("Rewards:", rewards)
    
    assert rewards[0] == 1.0, "Knight check failed"
    assert rewards[1] == 1.0, "Rook unblocked check failed"
    assert rewards[2] == 0.0, "Rook blocked check failed (should be 0)"
    assert rewards[3] == 1.0, "Pawn check failed"
    assert rewards[4] == 0.0, "Non-check failed"
    
    logger.info("Check potential tests passed!")

if __name__ == "__main__":
    test_initialization()
    test_check_potential()
