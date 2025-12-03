
import numpy as np
from game3d.board.board import Board
from game3d.common.shared_types import PieceType

def check_piece_counts():
    board = Board()
    coords, types, colors = board.get_initial_setup()
    
    vectorslider_count = np.sum(types == PieceType.VECTORSLIDER)
    print(f"Total VECTORSLIDER count: {vectorslider_count}")
    
    white_vectorsliders = np.sum((types == PieceType.VECTORSLIDER) & (colors == 1))
    black_vectorsliders = np.sum((types == PieceType.VECTORSLIDER) & (colors == 2))
    
    print(f"White VECTORSLIDER count: {white_vectorsliders}")
    print(f"Black VECTORSLIDER count: {black_vectorsliders}")
    
    # Print locations
    vs_indices = np.where(types == PieceType.VECTORSLIDER)[0]
    for idx in vs_indices:
        print(f"VECTORSLIDER at {coords[idx]} (Color: {colors[idx]})")

if __name__ == "__main__":
    check_piece_counts()
