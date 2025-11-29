
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game3d.board.board import Board
from game3d.cache.manager import get_cache_manager
from game3d.common.shared_types import Color, PieceType, COORD_DTYPE, PIECE_TYPE_DTYPE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_priest_count_corruption():
    print("Testing priest count corruption with duplicate coordinates...")
    
    # 1. Setup Board (Empty)
    board = Board()
    # Clear board for clean test
    cache_manager = get_cache_manager(board)
    cache_manager.occupancy_cache.clear()
    
    print(f"Initial Priest count: {cache_manager.occupancy_cache.get_priest_count(Color.WHITE)}")
    
    # 2. Place a Priest
    priest_pos = np.array([1, 1, 1], dtype=COORD_DTYPE)
    cache_manager.occupancy_cache.set_position(priest_pos, np.array([PieceType.PRIEST.value, Color.WHITE.value]))
    
    print(f"Priest count after placement: {cache_manager.occupancy_cache.get_priest_count(Color.WHITE)}")
    if cache_manager.occupancy_cache.get_priest_count(Color.WHITE) != 1:
        print("FAILURE: Priest count should be 1")
        return

    # 3. Update the SAME square with duplicates in batch
    # We will "move" the priest to the same spot (no change) or just set it again
    # But we pass the coordinate TWICE in the batch
    
    coords = np.array([
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=COORD_DTYPE)
    
    # We are replacing the priest with... a priest (or anything)
    # Let's say we replace it with a Pawn
    pieces = np.array([
        [PieceType.PAWN.value, Color.WHITE.value],
        [PieceType.PAWN.value, Color.WHITE.value]
    ], dtype=PIECE_TYPE_DTYPE)
    
    print("Executing batch update with duplicates (replacing Priest with Pawn twice)...")
    cache_manager.occupancy_cache.batch_set_positions(coords, pieces)
    
    # Expected: Priest count should be 0 (since priest is gone)
    # If bug exists: Priest count might be -1 (decremented twice)
    
    count = cache_manager.occupancy_cache.get_priest_count(Color.WHITE)
    print(f"Priest count after update: {count}")
    
    if count < 0:
        print("SUCCESS: Reproduced priest count corruption! Count is negative.")
    elif count != 0:
         print(f"FAILURE: Priest count is {count}, expected 0.")
    else:
        print("FAILURE: Priest count is 0 as expected. No corruption with this method.")

    # 4. Test removing priest with duplicates
    cache_manager.occupancy_cache.clear()
    cache_manager.occupancy_cache.set_position(priest_pos, np.array([PieceType.PRIEST.value, Color.WHITE.value]))
    print(f"\nReset. Priest count: {cache_manager.occupancy_cache.get_priest_count(Color.WHITE)}")
    
    # Remove priest (set to empty) with duplicates
    pieces_empty = np.array([
        [0, 0],
        [0, 0]
    ], dtype=PIECE_TYPE_DTYPE)
    
    print("Executing batch update with duplicates (removing Priest twice)...")
    cache_manager.occupancy_cache.batch_set_positions(coords, pieces_empty)
    
    count = cache_manager.occupancy_cache.get_priest_count(Color.WHITE)
    print(f"Priest count after removal: {count}")
    
    if count < 0:
        print("SUCCESS: Reproduced priest count corruption! Count is negative.")

if __name__ == "__main__":
    test_priest_count_corruption()
