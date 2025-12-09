
import numpy as np
import logging
from game3d.game.gamestate import GameState
from game3d.board.board import Board
from game3d.common.shared_types import PieceType, Color
from game3d.core.api import generate_legal_moves
from game3d.movement.generator import generate_legal_moves as generate_legal_moves_oo

# Setup logging
logging.basicConfig(level=logging.DEBUG)

def reproduce():
    # Use empty board as base
    gs = GameState(Board.empty(), Color.WHITE)
    
    # Access OccupancyCache directly
    occ = gs.cache_manager.occupancy_cache
    occ.clear()
    
    # Place White King at center (4,4,4)
    coord_k = np.array([4, 4, 4], dtype=np.int16)
    data_k = np.array([PieceType.KING.value, int(Color.WHITE)], dtype=np.int8)
    occ.set_position(coord_k, data_k)
    
    # Place White Priest safe (0,0,0)
    coord_p = np.array([0, 0, 0], dtype=np.int16)
    data_p = np.array([PieceType.PRIEST.value, int(Color.WHITE)], dtype=np.int8)
    occ.set_position(coord_p, data_p)
    
    # Place Black Rook attacking King at (4,4,8)
    coord_r = np.array([4, 4, 8], dtype=np.int16)
    data_r = np.array([PieceType.ROOK.value, int(Color.BLACK)], dtype=np.int8)
    occ.set_position(coord_r, data_r)
    
    # Update cache
    gs.cache_manager.move_cache.invalidate()
    
    print("Generating moves...")
    
    # 1. Test OO Generator (which applies high-level filters)
    moves = generate_legal_moves_oo(gs)
    print(f"Moves found (OO): {moves.shape[0]}")
    
    if moves.shape[0] == 0:
        print("FAIL: No moves found despite Priest presence!")
    else:
        print("SUCCESS: Moves found.")

if __name__ == "__main__":
    reproduce()
