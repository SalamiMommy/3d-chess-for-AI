
import numpy as np
from game3d.board.board import Board
from game3d.common.shared_types import PieceType

def check_specific_squares():
    board = Board()
    coords, types, colors = board.get_initial_setup()
    
    # Check (3, 2, 0)
    idx1 = np.where((coords[:, 0] == 3) & (coords[:, 1] == 2) & (coords[:, 2] == 0))[0]
    if len(idx1) > 0:
        t1 = types[idx1[0]]
        print(f"Piece at (3, 2, 0): {t1} ({PieceType(t1).name})")
    else:
        print("Piece at (3, 2, 0): None")
        
    # Check (5, 6, 1)
    idx2 = np.where((coords[:, 0] == 5) & (coords[:, 1] == 6) & (coords[:, 2] == 1))[0]
    if len(idx2) > 0:
        t2 = types[idx2[0]]
        print(f"Piece at (5, 6, 1): {t2} ({PieceType(t2).name})")
    else:
        print("Piece at (5, 6, 1): None")

if __name__ == "__main__":
    check_specific_squares()
