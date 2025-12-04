
import numpy as np
import os
import sys
from unittest.mock import MagicMock

# Mock numba before it gets imported by piece modules
numba_mock = MagicMock()
# Create a dummy decorator that returns the function as-is
def dummy_decorator(*args, **kwargs):
    def wrapper(func):
        return func
    return wrapper
# Handle both @njit and @njit(...)
def njit_mock(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return dummy_decorator

numba_mock.njit = njit_mock
numba_mock.objmode = MagicMock()
numba_mock.prange = range # Mock prange as normal range
sys.modules['numba'] = numba_mock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, PieceType, SIZE_MINUS_1
from game3d.pieces.pieces.xyqueen import _XY_SLIDER_DIRS, _KING_3D_DIRS
from game3d.pieces.pieces.xzqueen import _XZ_SLIDER_DIRS
from game3d.pieces.pieces.yzqueen import _YZ_SLIDER_DIRS
from game3d.pieces.pieces.vectorslider import VECTOR_DIRECTIONS
from game3d.pieces.pieces.facecone import FACE_CONE_MOVEMENT_VECTORS
from game3d.pieces.pieces.xzzigzag import XZ_ZIGZAG_DIRECTIONS
from game3d.pieces.pieces.yzzigzag import YZ_ZIGZAG_DIRECTIONS
from game3d.pieces.pieces.trigonalbishop import TRIGONAL_BISHOP_VECTORS
from game3d.pieces.pieces.bishop import BISHOP_MOVEMENT_VECTORS
from game3d.pieces.pieces.rook import ROOK_MOVEMENT_VECTORS
from game3d.pieces.pieces.queen import QUEEN_MOVEMENT_VECTORS
from game3d.pieces.pieces.trailblazer import ROOK_DIRECTIONS as TRAILBLAZER_DIRECTIONS
from game3d.pieces.pieces.spiral import SPIRAL_MOVEMENT_VECTORS
from game3d.pieces.pieces.reflector import _REFLECTOR_DIRS

# Configuration for each piece type
# Tuple format: (directions, max_distance)
# List of tuples allows combining different movement types (e.g. slider + king)
PIECE_CONFIGS = {
    PieceType.XYQUEEN: [
        (_XY_SLIDER_DIRS, 8),
        (_KING_3D_DIRS, 1)
    ],
    PieceType.XZQUEEN: [
        (_XZ_SLIDER_DIRS, 8),
        (_KING_3D_DIRS, 1)
    ],
    PieceType.YZQUEEN: [
        (_YZ_SLIDER_DIRS, 8),
        (_KING_3D_DIRS, 1)
    ],
    PieceType.VECTORSLIDER: [
        (VECTOR_DIRECTIONS, 8)
    ],
    PieceType.CONESLIDER: [
        (FACE_CONE_MOVEMENT_VECTORS, SIZE_MINUS_1)
    ],
    PieceType.XZZIGZAG: [
        (XZ_ZIGZAG_DIRECTIONS, 16) # Max steps 16 as per file
    ],
    PieceType.YZZIGZAG: [
        (YZ_ZIGZAG_DIRECTIONS, 16) # Max steps 16 as per file
    ],
    PieceType.TRIGONALBISHOP: [
        (TRIGONAL_BISHOP_VECTORS, SIZE_MINUS_1)
    ],
    PieceType.BISHOP: [
        (BISHOP_MOVEMENT_VECTORS, SIZE_MINUS_1)
    ],
    PieceType.ROOK: [
        (ROOK_MOVEMENT_VECTORS, 8)
    ],
    PieceType.QUEEN: [
        (QUEEN_MOVEMENT_VECTORS, 8)
    ],
    PieceType.TRAILBLAZER: [
        (TRAILBLAZER_DIRECTIONS, 4) # Using max buffed distance (4) to be safe, runtime will limit to 3 if unbuffed
    ],
    PieceType.SPIRAL: [
        (SPIRAL_MOVEMENT_VECTORS, 8)
    ],
    PieceType.REFLECTOR: [
        (_REFLECTOR_DIRS, SIZE_MINUS_1) # Special handling for bounces
    ]
}

def precompute_slider_rays():
    output_dir = os.path.join(os.path.dirname(__file__), "precomputed")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Precomputing slider rays in {output_dir}...")
    
    for piece_type, configs in PIECE_CONFIGS.items():
        print(f"Processing {piece_type.name}...")
        
        # We will store rays as a flat array of coordinates
        # rays_flat: [x1, y1, z1, x2, y2, z2, ...]
        # ray_offsets: [start_idx_ray_0, start_idx_ray_1, ...]
        # square_offsets: [start_idx_sq_0, start_idx_sq_1, ...]
        
        rays_flat_list = []
        ray_offsets_list = [0]
        square_offsets_list = [0]
        
        current_ray_offset_idx = 0
        
        # Iterate over all squares
        for z in range(SIZE):
            for y in range(SIZE):
                for x in range(SIZE):
                    pos = np.array([x, y, z], dtype=COORD_DTYPE)
                    
                    # For this square, generate all rays
                    for directions, max_dist in configs:
                        
                        # Special handling for Reflector (bounces)
                        if piece_type == PieceType.REFLECTOR:
                             # Reflector logic is complex (bounces). 
                             # We simulate the bounces here.
                             for d in directions:
                                ray_coords = []
                                curr_pos = pos.copy()
                                curr_dir = d.copy()
                                bounces = 0
                                max_bounces = 2
                                
                                # Trace ray
                                for _ in range(24): # Max path length safety
                                    next_pos = curr_pos + curr_dir
                                    
                                    # Bounds check
                                    out_x = next_pos[0] < 0 or next_pos[0] >= SIZE
                                    out_y = next_pos[1] < 0 or next_pos[1] >= SIZE
                                    out_z = next_pos[2] < 0 or next_pos[2] >= SIZE
                                    
                                    if out_x or out_y or out_z:
                                        if bounces >= max_bounces:
                                            break
                                        
                                        # Reflect
                                        if out_x: curr_dir[0] = -curr_dir[0]
                                        if out_y: curr_dir[1] = -curr_dir[1]
                                        if out_z: curr_dir[2] = -curr_dir[2]
                                        
                                        bounces += 1
                                        continue
                                    
                                    # Valid step
                                    ray_coords.append(next_pos.copy())
                                    curr_pos = next_pos
                                
                                if ray_coords:
                                    rays_flat_list.extend(ray_coords)
                                    # Update ray offset (points to END of this ray in flat list)
                                    ray_offsets_list.append(len(rays_flat_list))
                                    current_ray_offset_idx += 1

                        else:
                            # Standard linear rays
                            for d in directions:
                                ray_coords = []
                                curr_pos = pos.copy()
                                
                                for _ in range(max_dist):
                                    curr_pos += d
                                    
                                    if (curr_pos[0] < 0 or curr_pos[0] >= SIZE or
                                        curr_pos[1] < 0 or curr_pos[1] >= SIZE or
                                        curr_pos[2] < 0 or curr_pos[2] >= SIZE):
                                        break
                                    
                                    ray_coords.append(curr_pos.copy())
                                
                                if ray_coords:
                                    rays_flat_list.extend(ray_coords)
                                    ray_offsets_list.append(len(rays_flat_list))
                                    current_ray_offset_idx += 1
                    
                    # End of square
                    square_offsets_list.append(current_ray_offset_idx)
        
        # Convert to numpy arrays
        # rays_flat: (N, 3) coords
        if rays_flat_list:
            rays_flat = np.array(rays_flat_list, dtype=COORD_DTYPE)
        else:
            rays_flat = np.empty((0, 3), dtype=COORD_DTYPE)
            
        # ray_offsets: indices into rays_flat. 
        # Note: We want start/end pairs. 
        # ray_offsets_list contains cumulative lengths.
        # ray_offsets[i] is start of ray i. ray_offsets[i+1] is end.
        ray_offsets = np.array(ray_offsets_list, dtype=np.int32)
        
        # square_offsets: indices into ray_offsets.
        # square_offsets[i] is start ray index for square i.
        square_offsets = np.array(square_offsets_list, dtype=np.int32)
        
        # Save
        np.save(os.path.join(output_dir, f"rays_{piece_type.name}_flat.npy"), rays_flat)
        np.save(os.path.join(output_dir, f"rays_{piece_type.name}_ray_offsets.npy"), ray_offsets)
        np.save(os.path.join(output_dir, f"rays_{piece_type.name}_sq_offsets.npy"), square_offsets)
        
        print(f"  Saved {rays_flat.shape[0]} coords, {len(ray_offsets)-1} rays")

if __name__ == "__main__":
    precompute_slider_rays()
