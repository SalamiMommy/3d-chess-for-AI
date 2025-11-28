
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from game3d.common.shared_types import COORD_DTYPE, SIZE, SIZE_SQUARED, PieceType

# Define vectors locally to avoid import issues and private variable access
# King-like vectors (26 directions)
dx_vals, dy_vals, dz_vals = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
all_coords = np.stack([dx_vals.ravel(), dy_vals.ravel(), dz_vals.ravel()], axis=1)
origin_mask = np.any(all_coords != 0, axis=1)
KING_VECTORS = all_coords[origin_mask].astype(COORD_DTYPE)

# Knight vectors
KNIGHT_VECTORS = np.array([
    [1, 2, 0], [2, 1, 0], [-1, 2, 0], [-2, 1, 0],
    [1, -2, 0], [2, -1, 0], [-1, -2, 0], [-2, -1, 0],
    [0, 1, 2], [0, 2, 1], [0, -1, 2], [0, -2, 1],
    [0, 1, -2], [0, 2, -1], [0, -1, -2], [0, -2, -1],
    [1, 0, 2], [2, 0, 1], [-1, 0, 2], [-2, 0, 1],
    [1, 0, -2], [2, 0, -1], [-1, 0, -2], [-2, 0, -1]
], dtype=COORD_DTYPE)

# Knight31 vectors
KNIGHT31_VECTORS = np.array([
    (3, 1, 1), (3, 1, -1), (3, -1, 1), (3, -1, -1),
    (-3, 1, 1), (-3, 1, -1), (-3, -1, 1), (-3, -1, -1),
    (1, 3, 1), (1, 3, -1), (1, -3, 1), (1, -3, -1),
    (-1, 3, 1), (-1, 3, -1), (-1, -3, 1), (-1, -3, -1),
    (1, 1, 3), (1, 1, -3), (1, -1, 3), (1, -1, -3),
    (-1, 1, 3), (-1, 1, -3), (-1, -1, 3), (-1, -1, -3),
], dtype=COORD_DTYPE)

# Knight32 vectors
KNIGHT32_VECTORS = np.array([
    (3, 2, 2), (3, 2, -2), (3, -2, 2), (3, -2, -2),
    (-3, 2, 2), (-3, 2, -2), (-3, -2, 2), (-3, -2, -2),
    (2, 3, 2), (2, 3, -2), (2, -3, 2), (2, -3, -2),
    (-2, 3, 2), (-2, 3, -2), (-2, -3, 2), (-2, -3, -2),
    (2, 2, 3), (2, 2, -3), (2, -2, 3), (2, -2, -3),
    (-2, 2, 3), (-2, 2, -3), (-2, -2, 3), (-2, -2, -3),
], dtype=COORD_DTYPE)

# Wall vectors (orthogonal)
WALL_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

# Orbiter vectors (Manhattan distance 4)
ORBITER_MANHATTAN_DISTANCE = 4
ORBITER_VECTORS = np.array([
    (dx, dy, dz)
    for dx in range(-ORBITER_MANHATTAN_DISTANCE, ORBITER_MANHATTAN_DISTANCE + 1)
    for dy in range(-ORBITER_MANHATTAN_DISTANCE, ORBITER_MANHATTAN_DISTANCE + 1)
    for dz in range(-ORBITER_MANHATTAN_DISTANCE, ORBITER_MANHATTAN_DISTANCE + 1)
    if abs(dx) + abs(dy) + abs(dz) == ORBITER_MANHATTAN_DISTANCE
], dtype=COORD_DTYPE)

# Archer vectors (King + Radius 2 Surface)
# NOTE: Archer shots (radius 2) are capture-only and cannot be handled by standard jump engine precomputation
# which assumes move-or-capture. So we only precompute the movement part (King vectors).
ARCHER_COMBINED_VECTORS = KING_VECTORS

# Mapping of PieceType to Vectors
PIECE_VECTORS = {
    PieceType.KING: KING_VECTORS,
    PieceType.PRIEST: KING_VECTORS,
    PieceType.KNIGHT: KNIGHT_VECTORS,
    PieceType.KNIGHT31: KNIGHT31_VECTORS,
    PieceType.KNIGHT32: KNIGHT32_VECTORS,
    PieceType.BOMB: KING_VECTORS, # Bomb uses King vectors for movement
    PieceType.BLACKHOLE: KING_VECTORS,
    PieceType.WHITEHOLE: KING_VECTORS,
    PieceType.HIVE: KING_VECTORS,
    PieceType.SWAPPER: KING_VECTORS, # Swapper uses King vectors for movement
    PieceType.INFILTRATOR: KING_VECTORS, # Infiltrator uses King vectors for movement
    PieceType.ORBITER: ORBITER_VECTORS,
    PieceType.ARCHER: ARCHER_COMBINED_VECTORS,
    PieceType.FREEZER: KING_VECTORS,
    PieceType.SPEEDER: KING_VECTORS,
    PieceType.WALL: WALL_VECTORS,
    PieceType.SLOWER: KING_VECTORS, # Assumed King-like
    PieceType.GEOMANCER: KING_VECTORS, # Assumed King-like
    PieceType.ARMOUR: KING_VECTORS, # Assumed King-like
    PieceType.NEBULA: KING_VECTORS, # Assumed King-like
    PieceType.ECHO: KING_VECTORS, # Assumed King-like
    PieceType.PANEL: KING_VECTORS, # Assumed King-like
    PieceType.MIRROR: KING_VECTORS, # Assumed King-like
    PieceType.FRIENDLYTELEPORTER: KING_VECTORS, # Assumed King-like
}

def precompute_moves():
    output_dir = os.path.join(os.path.dirname(__file__), "precomputed")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Precomputing moves in {output_dir}...")
    
    for piece_type, vectors in PIECE_VECTORS.items():
        print(f"Processing {piece_type.name}...")
        
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
        
        # Save to file
        filename = f"moves_{piece_type.name}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, all_moves)
        print(f"Saved {filename}")

if __name__ == "__main__":
    precompute_moves()
