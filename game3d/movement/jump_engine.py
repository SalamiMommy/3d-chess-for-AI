
import numpy as np
from numba import njit
from game3d.common.shared_types import SIZE, MOVE_DTYPE, COLOR_EMPTY, COORD_DTYPE

@njit(cache=True)
def _generate_and_filter_jump_moves_batch(
    positions, vectors, board_color, can_capture, own_color
):
    """
    Generate jump moves for a batch of pieces.
    
    Args:
        positions: (N, 3) coordinates of pieces.
        vectors: (M, 3) jump vectors.
        board_color: (SIZE, SIZE, SIZE) board color array.
        can_capture: bool, if True captures are allowed.
        own_color: int, color of the moving pieces.
        
    Returns:
        (K, 6) array of moves.
    """
    n_pieces = positions.shape[0]
    n_vectors = vectors.shape[0]
    max_moves = n_pieces * n_vectors
    
    out_moves = np.empty((max_moves, 6), dtype=COORD_DTYPE)
    count = 0
    
    for i in range(n_pieces):
        px, py, pz = positions[i]
        
        for k in range(n_vectors):
            vx, vy, vz = vectors[k]
            
            tx = px + vx
            ty = py + vy
            tz = pz + vz
            
            # Bounds check
            if tx < 0 or tx >= SIZE or ty < 0 or ty >= SIZE or tz < 0 or tz >= SIZE:
                continue
                
            target_color = board_color[tx, ty, tz]
            
            # Check occupancy
            if target_color == COLOR_EMPTY:
                # Quiet move
                out_moves[count, 0] = px
                out_moves[count, 1] = py
                out_moves[count, 2] = pz
                out_moves[count, 3] = tx
                out_moves[count, 4] = ty
                out_moves[count, 5] = tz
                count += 1
            elif can_capture and target_color != own_color:
                # Capture
                out_moves[count, 0] = px
                out_moves[count, 1] = py
                out_moves[count, 2] = pz
                out_moves[count, 3] = tx
                out_moves[count, 4] = ty
                out_moves[count, 5] = tz
                count += 1
                
    return out_moves[:count]
