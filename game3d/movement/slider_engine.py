
import numpy as np
from numba import njit
from game3d.common.shared_types import SIZE, MOVE_DTYPE, COLOR_EMPTY, COORD_DTYPE

@njit(cache=True)
def _generate_all_slider_moves_batch(
    own_color, positions, vectors, max_dists, board_color, capture_only_flag_ignored=False
):
    """
    Generate slider moves for a batch of pieces (e.g. Rook, Bishop, Queen).
    
    Args:
        own_color: int, color of the moving pieces.
        positions: (N, 3) coordinates.
        vectors: (M, 3) direction vectors.
        max_dists: (N,) max distance for each piece (usually SIZE-1 or modified range).
        board_color: (SIZE, SIZE, SIZE) color array.
        capture_only_flag_ignored: bool (ignored/placeholder to match signature).
        
    Returns:
        tuple (moves, None)
        moves: (K, 6) array.
    """
    n_pieces = positions.shape[0]
    n_vectors = vectors.shape[0]
    
    # Heuristic size estimation: N * M * Avg_Dist(3)
    estimated_size = n_pieces * n_vectors * 4 
    if estimated_size < 1024: estimated_size = 1024
    
    out_moves = np.empty((estimated_size, 6), dtype=COORD_DTYPE)
    count = 0
    
    for i in range(n_pieces):
        px, py, pz = positions[i]
        limit = max_dists[i]
        
        for k in range(n_vectors):
            vx, vy, vz = vectors[k]
            
            for dist in range(1, limit + 1):
                tx = px + vx * dist
                ty = py + vy * dist
                tz = pz + vz * dist
                
                # Bounds check
                if tx < 0 or tx >= SIZE or ty < 0 or ty >= SIZE or tz < 0 or tz >= SIZE:
                    break
                    
                target_color = board_color[tx, ty, tz]
                
                if target_color == COLOR_EMPTY:
                    # Legal quiet move
                    if count >= out_moves.shape[0]:
                        # Resize
                        new_size = out_moves.shape[0] * 2
                        new_arr = np.empty((new_size, 6), dtype=COORD_DTYPE)
                        new_arr[:count] = out_moves[:count]
                        out_moves = new_arr
                        
                    out_moves[count, 0] = px
                    out_moves[count, 1] = py
                    out_moves[count, 2] = pz
                    out_moves[count, 3] = tx
                    out_moves[count, 4] = ty
                    out_moves[count, 5] = tz
                    count += 1
                else:
                    # Occupied
                    if target_color != own_color:
                        # Capture
                        if count >= out_moves.shape[0]:
                            new_size = out_moves.shape[0] * 2
                            new_arr = np.empty((new_size, 6), dtype=COORD_DTYPE)
                            new_arr[:count] = out_moves[:count]
                            out_moves = new_arr
                            
                        out_moves[count, 0] = px
                        out_moves[count, 1] = py
                        out_moves[count, 2] = pz
                        out_moves[count, 3] = tx
                        out_moves[count, 4] = ty
                        out_moves[count, 5] = tz
                        count += 1
                    
                    # Blocked irrespective of color
                    break
                    
    return out_moves[:count], None
