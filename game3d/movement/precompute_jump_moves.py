
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
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, PieceType

# Import vectors directly from piece definitions to ensure consistency
from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS, BUFFED_KING_MOVEMENT_VECTORS
from game3d.pieces.pieces.knight import KNIGHT_MOVEMENT_VECTORS, BUFFED_KNIGHT_MOVEMENT_VECTORS
from game3d.pieces.pieces.orbiter import ORBITER_MOVEMENT_VECTORS, BUFFED_ORBITER_MOVEMENT_VECTORS
from game3d.pieces.pieces.nebula import NEBULA_MOVEMENT_VECTORS, BUFFED_NEBULA_MOVEMENT_VECTORS
from game3d.pieces.pieces.echo import ECHO_MOVEMENT_VECTORS, BUFFED_ECHO_MOVEMENT_VECTORS
from game3d.pieces.pieces.wall import UNBUFFED_WALL_VECTORS
from game3d.pieces.pieces.panel import PANEL_MOVEMENT_VECTORS, BUFFED_PANEL_MOVEMENT_VECTORS
from game3d.pieces.pieces.edgerook import EDGE_ROOK_VECTORS

# Knight31 vectors (3,1,0) - unbuffed (Local definition as files do not exist)
KNIGHT31_VECTORS = np.array([
    (3, 1, 0), (3, -1, 0), (-3, 1, 0), (-3, -1, 0),
    (1, 3, 0), (1, -3, 0), (-1, 3, 0), (-1, -3, 0),
    (3, 0, 1), (3, 0, -1), (-3, 0, 1), (-3, 0, -1),
    (0, 3, 1), (0, 3, -1), (0, -3, 1), (0, -3, -1),
    (1, 0, 3), (1, 0, -3), (-1, 0, 3), (-1, 0, -3),
    (0, 1, 3), (0, 1, -3), (0, -1, 3), (0, -1, -3),
], dtype=COORD_DTYPE)

# Buffed Knight31 (3,1,1) - buffed (Local definition as files do not exist)
BUFFED_KNIGHT31_VECTORS = np.array([
    (3, 1, 1), (3, 1, -1), (3, -1, 1), (3, -1, -1),
    (-3, 1, 1), (-3, 1, -1), (-3, -1, 1), (-3, -1, -1),
    (1, 3, 1), (1, 3, -1), (1, -3, 1), (1, -3, -1),
    (-1, 3, 1), (-1, 3, -1), (-1, -3, 1), (-1, -3, -1),
    (1, 1, 3), (1, 1, -3), (1, -1, 3), (1, -1, -3),
    (-1, 1, 3), (-1, 1, -3), (-1, -1, 3), (-1, -1, -3),
], dtype=COORD_DTYPE)

# Knight32 vectors (3,2,0) - unbuffed (Local definition as files do not exist)
KNIGHT32_VECTORS = np.array([
    (3, 2, 0), (3, -2, 0), (-3, 2, 0), (-3, -2, 0),
    (2, 3, 0), (2, -3, 0), (-2, 3, 0), (-2, -3, 0),
    (3, 0, 2), (3, 0, -2), (-3, 0, 2), (-3, 0, -2),
    (0, 3, 2), (0, 3, -2), (0, -3, 2), (0, -3, -2),
    (2, 0, 3), (2, 0, -3), (-2, 0, 3), (-2, 0, -3),
    (0, 2, 3), (0, 2, -3), (0, -2, 3), (0, -2, -3),
], dtype=COORD_DTYPE)

# Buffed Knight32 (3,2,1) - buffed (Local definition as files do not exist)
BUFFED_KNIGHT32_VECTORS = np.array([
    (3, 2, 1), (3, 2, -1), (3, -2, 1), (3, -2, -1),
    (-3, 2, 1), (-3, 2, -1), (-3, -2, 1), (-3, -2, -1),
    (2, 3, 1), (2, 3, -1), (2, -3, 1), (2, -3, -1),
    (-2, 3, 1), (-2, 3, -1), (-2, -3, 1), (-2, -3, -1),
    (2, 1, 3), (2, 1, -3), (2, -1, 3), (2, -1, -3),
    (-2, 1, 3), (-2, 1, -3), (-2, -1, 3), (-2, -1, -3),
    (1, 2, 3), (1, 2, -3), (1, -2, 3), (1, -2, -3),
    (-1, 2, 3), (-1, 2, -3), (-1, -2, 3), (-1, -2, -3),
    (1, 3, 2), (1, 3, -2), (1, -3, 2), (1, -3, -2),
    (-1, 3, 2), (-1, 3, -2), (-1, -3, 2), (-1, -3, -2),
    (3, 1, 2), (3, 1, -2), (3, -1, 2), (3, -1, -2),
    (-3, 1, 2), (-3, 1, -2), (-3, -1, 2), (-3, -1, -2),
], dtype=COORD_DTYPE)

# Archer vectors (King + Radius 2 Surface)
# NOTE: Archer shots (radius 2) are capture-only and cannot be handled by standard jump engine precomputation
# which assumes move-or-capture. So we only precompute the movement part (King vectors).
ARCHER_COMBINED_VECTORS = KING_MOVEMENT_VECTORS

# Mapping of PieceType to Vectors - now supporting both unbuffed and buffed
PIECE_VECTORS = {
    PieceType.KING: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.PRIEST: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.KNIGHT: {'unbuffed': KNIGHT_MOVEMENT_VECTORS, 'buffed': BUFFED_KNIGHT_MOVEMENT_VECTORS},
    PieceType.KNIGHT31: {'unbuffed': KNIGHT31_VECTORS, 'buffed': BUFFED_KNIGHT31_VECTORS},
    PieceType.KNIGHT32: {'unbuffed': KNIGHT32_VECTORS, 'buffed': BUFFED_KNIGHT32_VECTORS},
    PieceType.BOMB: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.BLACKHOLE: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.WHITEHOLE: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.HIVE: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.SWAPPER: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.INFILTRATOR: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.ORBITER: {'unbuffed': ORBITER_MOVEMENT_VECTORS, 'buffed': BUFFED_ORBITER_MOVEMENT_VECTORS},
    PieceType.ARCHER: {'unbuffed': ARCHER_COMBINED_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.FREEZER: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.SPEEDER: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.WALL: {'unbuffed': UNBUFFED_WALL_VECTORS, 'buffed': UNBUFFED_WALL_VECTORS},  # Wall doesn't have buffed variant
    PieceType.SLOWER: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.GEOMANCER: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.ARMOUR: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.NEBULA: {'unbuffed': NEBULA_MOVEMENT_VECTORS, 'buffed': BUFFED_NEBULA_MOVEMENT_VECTORS},
    PieceType.ECHO: {'unbuffed': ECHO_MOVEMENT_VECTORS, 'buffed': BUFFED_ECHO_MOVEMENT_VECTORS},
    PieceType.PANEL: {'unbuffed': PANEL_MOVEMENT_VECTORS, 'buffed': BUFFED_PANEL_MOVEMENT_VECTORS},
    PieceType.MIRROR: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.FRIENDLYTELEPORTER: {'unbuffed': KING_MOVEMENT_VECTORS, 'buffed': BUFFED_KING_MOVEMENT_VECTORS},
    PieceType.EDGEROOK: {'unbuffed': EDGE_ROOK_VECTORS, 'buffed': EDGE_ROOK_VECTORS},
}

def precompute_moves():
    output_dir = os.path.join(os.path.dirname(__file__), "precomputed")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Precomputing moves in {output_dir}...")
    
    for piece_type, vector_dict in PIECE_VECTORS.items():
        print(f"Processing {piece_type.name}...")
        
        # Process both unbuffed and buffed variants
        for variant in ['unbuffed', 'buffed']:
            vectors = vector_dict[variant]
            print(f"  - {variant} variant ({len(vectors)} directions)...")
            
            # Array to store moves for each position
            # Size is SIZE^3
            all_moves = np.empty(SIZE_SQUARED * SIZE, dtype=object)
            
            for z in range(SIZE):
                for y in range(SIZE):
                    for x in range(SIZE):
                        flat_idx = x + SIZE * y + SIZE_SQUARED * z
                        pos = np.array([x, y, z], dtype=COORD_DTYPE)
                        
                        # Calculate targets
                        targets = pos + vectors
                        
                        # Filter bounds
                        valid_mask = (
                            (targets[:, 0] >= 0) & (targets[:, 0] < SIZE) &
                            (targets[:, 1] >= 0) & (targets[:, 1] < SIZE) &
                            (targets[:, 2] >= 0) & (targets[:, 2] < SIZE)
                        )
                        
                        valid_targets = targets[valid_mask]
                        
                        # Store
                        all_moves[flat_idx] = valid_targets
            
            # Save legacy object array
            filename = f"moves_{piece_type.name}_{variant}.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, all_moves)
            print(f"  Saved {filename}")

            # Generate and save flat arrays for Numba
            # 1. Calculate total moves and offsets
            n_positions = SIZE_SQUARED * SIZE
            offsets = np.zeros(n_positions + 1, dtype=np.int32)
            
            current_offset = 0
            flat_moves_list = []
            
            for i in range(n_positions):
                offsets[i] = current_offset
                moves = all_moves[i]
                if moves is not None and moves.size > 0:
                    current_offset += moves.shape[0]
                    flat_moves_list.append(moves)
            
            offsets[n_positions] = current_offset
            
            # 2. Create flat moves array
            if flat_moves_list:
                flat_moves = np.concatenate(flat_moves_list, axis=0)
            else:
                flat_moves = np.empty((0, 3), dtype=COORD_DTYPE)
                
            # 3. Save flat files
            flat_filename = f"moves_{piece_type.name}_{variant}_flat.npy"
            flat_filepath = os.path.join(output_dir, flat_filename)
            np.save(flat_filepath, flat_moves)
            print(f"  Saved {flat_filename}")
            
            offsets_filename = f"moves_{piece_type.name}_{variant}_offsets.npy"
            offsets_filepath = os.path.join(output_dir, offsets_filename)
            np.save(offsets_filepath, offsets)
            print(f"  Saved {offsets_filename}")


if __name__ == "__main__":
    precompute_moves()
