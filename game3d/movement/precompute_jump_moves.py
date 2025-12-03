
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, PieceType

# Define vectors locally to avoid import issues and private variable access
# King-like vectors (26 directions) - unbuffed
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
origin_mask = np.any(all_coords != 0, axis=1)
KING_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

# Buffed King: 5x5x5 cube (Chebyshev distance 2)
dx_vals_b, dy_vals_b, dz_vals_b = np.meshgrid(
    [-2, -1, 0, 1, 2], 
    [-2, -1, 0, 1, 2], 
    [-2, -1, 0, 1, 2], 
    indexing='ij'
)
all_coords_b = np.stack([dx_vals_b.ravel(), dy_vals_b.ravel(), dz_vals_b.ravel()], axis=1)
origin_mask_b = np.any(all_coords_b != 0, axis=1)
BUFFED_KING_VECTORS = all_coords_b[origin_mask_b].astype(COORD_DTYPE)

# Knight vectors (2,1,0) - unbuffed
KNIGHT_VECTORS = np.array([
    [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
    [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0],
    [0, 1, 2], [0, 2, 1], [0, -1, 2], [0, -2, 1],
    [0, 1, -2], [0, 2, -1], [0, -1, -2], [0, -2, -1],
    [1, 0, 2], [2, 0, 1], [-1, 0, 2], [-2, 0, 1],
    [1, 0, -2], [2, 0, -1], [-1, 0, -2], [-2, 0, -1]
], dtype=COORD_DTYPE)

# Buffed knight movement vectors (2,1,1) - buffed
BUFFED_KNIGHT_VECTORS = np.array([
    [2, 1, 1], [2, 1, -1], [2, -1, 1], [2, -1, -1],
    [-2, 1, 1], [-2, 1, -1], [-2, -1, 1], [-2, -1, -1],
    [1, 2, 1], [1, 2, -1], [1, -2, 1], [1, -2, -1],
    [-1, 2, 1], [-1, 2, -1], [-1, -2, 1], [-1, -2, -1],
    [1, 1, 2], [1, 1, -2], [1, -1, 2], [1, -1, -2],
    [-1, 1, 2], [-1, 1, -2], [-1, -1, 2], [-1, -1, -2],
], dtype=COORD_DTYPE)

# Knight31 vectors (3,1,0) - unbuffed
KNIGHT31_VECTORS = np.array([
    (3, 1, 0), (3, -1, 0), (-3, 1, 0), (-3, -1, 0),
    (1, 3, 0), (1, -3, 0), (-1, 3, 0), (-1, -3, 0),
    (3, 0, 1), (3, 0, -1), (-3, 0, 1), (-3, 0, -1),
    (0, 3, 1), (0, 3, -1), (0, -3, 1), (0, -3, -1),
    (1, 0, 3), (1, 0, -3), (-1, 0, 3), (-1, 0, -3),
    (0, 1, 3), (0, 1, -3), (0, -1, 3), (0, -1, -3),
], dtype=COORD_DTYPE)

# Buffed Knight31 (3,1,1) - buffed
BUFFED_KNIGHT31_VECTORS = np.array([
    (3, 1, 1), (3, 1, -1), (3, -1, 1), (3, -1, -1),
    (-3, 1, 1), (-3, 1, -1), (-3, -1, 1), (-3, -1, -1),
    (1, 3, 1), (1, 3, -1), (1, -3, 1), (1, -3, -1),
    (-1, 3, 1), (-1, 3, -1), (-1, -3, 1), (-1, -3, -1),
    (1, 1, 3), (1, 1, -3), (1, -1, 3), (1, -1, -3),
    (-1, 1, 3), (-1, 1, -3), (-1, -1, 3), (-1, -1, -3),
], dtype=COORD_DTYPE)

# Knight32 vectors (3,2,0) - unbuffed
KNIGHT32_VECTORS = np.array([
    (3, 2, 0), (3, -2, 0), (-3, 2, 0), (-3, -2, 0),
    (2, 3, 0), (2, -3, 0), (-2, 3, 0), (-2, -3, 0),
    (3, 0, 2), (3, 0, -2), (-3, 0, 2), (-3, 0, -2),
    (0, 3, 2), (0, 3, -2), (0, -3, 2), (0, -3, -2),
    (2, 0, 3), (2, 0, -3), (-2, 0, 3), (-2, 0, -3),
    (0, 2, 3), (0, 2, -3), (0, -2, 3), (0, -2, -3),
], dtype=COORD_DTYPE)

# Buffed Knight32 (3,2,1) - buffed
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

# Wall vectors (orthogonal)
WALL_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

# Orbiter vectors - Radius 3 Euclidean sphere (surface only) - unbuffed
ORBITER_VECTORS = np.array([
    (dx, dy, dz)
    for dx in range(-4, 5)
    for dy in range(-4, 5)
    for dz in range(-4, 5)
    if 8 <= (dx*dx + dy*dy + dz*dz) <= 11
], dtype=COORD_DTYPE)

# Buffed Orbiter - Radius 4 Euclidean sphere (surface only) - buffed
BUFFED_ORBITER_VECTORS = np.array([
    (dx, dy, dz)
    for dx in range(-5, 6)
    for dy in range(-5, 6)
    for dz in range(-5, 6)
    if 14 <= (dx*dx + dy*dy + dz*dz) <= 18
], dtype=COORD_DTYPE)

# Nebula - All positions within radius 2 (excluding origin) - unbuffed
NEBULA_VECTORS = np.array([
    (dx, dy, dz)
    for dx in range(-2, 3)
    for dy in range(-2, 3)
    for dz in range(-2, 3)
    if 0 < (dx*dx + dy*dy + dz*dz) <= 4
], dtype=COORD_DTYPE)

# Buffed Nebula - All positions within radius 3 (excluding origin) - buffed
BUFFED_NEBULA_VECTORS = np.array([
    (dx, dy, dz)
    for dx in range(-3, 4)
    for dy in range(-3, 4)
    for dz in range(-3, 4)
    if 0 < (dx*dx + dy*dy + dz*dz) <= 9
], dtype=COORD_DTYPE)

# Panel vectors - 3x3 panels at distance 2 - unbuffed
def _create_panel_vectors(distance):
    vectors = []
    r = [-1, 0, 1]
    # X faces
    for x in [-distance, distance]:
        for y in r:
            for z in r:
                vectors.append([x, y, z])
    # Y faces
    for y in [-distance, distance]:
        for x in r:
            for z in r:
                vectors.append([x, y, z])
    # Z faces
    for z in [-distance, distance]:
        for x in r:
            for y in r:
                vectors.append([x, y, z])
    return np.array(vectors, dtype=COORD_DTYPE)

PANEL_VECTORS = _create_panel_vectors(2)
BUFFED_PANEL_VECTORS = _create_panel_vectors(3)

# Echo piece - 6 cardinal anchors at offset 2 + 26 radius-1 bubble offsets - unbuffed
# Build radius 1 offsets
_radius_1_coords = np.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T
_radius_1_offsets = _radius_1_coords[np.sum(_radius_1_coords * _radius_1_coords, axis=1) <= 1].astype(COORD_DTYPE)

_ECHO_ANCHORS = np.array([
    [-2, 0, 0], [2, 0, 0],
    [0, -2, 0], [0, 2, 0],
    [0, 0, -2], [0, 0, 2]
], dtype=COORD_DTYPE)
ECHO_VECTORS = (_ECHO_ANCHORS[:, None, :] + _radius_1_offsets[None, :, :]).reshape(-1, 3)

# Buffed Echo - 6 cardinal anchors at offset 3 + 26 radius-1 bubble offsets - buffed
_BUFFED_ECHO_ANCHORS = np.array([
    [-3, 0, 0], [3, 0, 0],
    [0, -3, 0], [0, 3, 0],
    [0, 0, -3], [0, 0, 3]
], dtype=COORD_DTYPE)
BUFFED_ECHO_VECTORS = (_BUFFED_ECHO_ANCHORS[:, None, :] + _radius_1_offsets[None, :, :]).reshape(-1, 3)

# Archer vectors (King + Radius 2 Surface)
# NOTE: Archer shots (radius 2) are capture-only and cannot be handled by standard jump engine precomputation
# which assumes move-or-capture. So we only precompute the movement part (King vectors).
ARCHER_COMBINED_VECTORS = KING_VECTORS

# Mapping of PieceType to Vectors - now supporting both unbuffed and buffed
PIECE_VECTORS = {
    PieceType.KING: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.PRIEST: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.KNIGHT: {'unbuffed': KNIGHT_VECTORS, 'buffed': BUFFED_KNIGHT_VECTORS},
    PieceType.KNIGHT31: {'unbuffed': KNIGHT31_VECTORS, 'buffed': BUFFED_KNIGHT31_VECTORS},
    PieceType.KNIGHT32: {'unbuffed': KNIGHT32_VECTORS, 'buffed': BUFFED_KNIGHT32_VECTORS},
    PieceType.BOMB: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.BLACKHOLE: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.WHITEHOLE: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.HIVE: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.SWAPPER: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.INFILTRATOR: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.ORBITER: {'unbuffed': ORBITER_VECTORS, 'buffed': BUFFED_ORBITER_VECTORS},
    PieceType.ARCHER: {'unbuffed': ARCHER_COMBINED_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.FREEZER: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.SPEEDER: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.WALL: {'unbuffed': WALL_VECTORS, 'buffed': WALL_VECTORS},  # Wall doesn't have buffed variant
    PieceType.SLOWER: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.GEOMANCER: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.ARMOUR: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.NEBULA: {'unbuffed': NEBULA_VECTORS, 'buffed': BUFFED_NEBULA_VECTORS},
    PieceType.ECHO: {'unbuffed': ECHO_VECTORS, 'buffed': BUFFED_ECHO_VECTORS},
    PieceType.PANEL: {'unbuffed': PANEL_VECTORS, 'buffed': BUFFED_PANEL_VECTORS},
    PieceType.MIRROR: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
    PieceType.FRIENDLYTELEPORTER: {'unbuffed': KING_VECTORS, 'buffed': BUFFED_KING_VECTORS},
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
