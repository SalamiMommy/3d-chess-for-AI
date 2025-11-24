
import numpy as np
import os
from game3d.common.shared_types import SIZE, COORD_DTYPE, PieceType, RADIUS_2_OFFSETS, RADIUS_3_OFFSETS
from game3d.common.coord_utils import in_bounds_vectorized

# Import piece modules to access vectors
from game3d.pieces.pieces.knight import KNIGHT_MOVEMENT_VECTORS
from game3d.pieces.pieces.kinglike import KING_MOVEMENT_VECTORS
from game3d.pieces.pieces.bigknights import KNIGHT31_MOVEMENT_VECTORS, KNIGHT32_MOVEMENT_VECTORS
from game3d.pieces.pieces.orbiter import _ORBITAL_DIRS
from game3d.pieces.pieces.echo import _ECHO_DIRECTIONS
from game3d.pieces.pieces.panel import PANEL_MOVEMENT_VECTORS
from game3d.pieces.pieces.edgerook import EDGE_ROOK_VECTORS, _EDGE_NEIGHBORS
from game3d.pieces.pieces.freezer import FREEZER_MOVEMENT_VECTORS
from game3d.pieces.pieces.wall import WALL_MOVEMENT_VECTORS
from game3d.pieces.pieces.bomb import BOMB_MOVEMENT_VECTORS
from game3d.pieces.pieces.armour import ARMOUR_MOVEMENT_VECTORS
from game3d.pieces.pieces.slower import SLOWER_MOVEMENT_VECTORS
from game3d.pieces.pieces.swapper import _SWAPPER_MOVEMENT_VECTORS
from game3d.pieces.pieces.hive import HIVE_DIRECTIONS_3D
from game3d.pieces.pieces.blackhole import BLACKHOLE_MOVEMENT_VECTORS

# Output directory
OUTPUT_DIR = "/home/salamimommy/Documents/code/3d-chess-for-AI/game3d/movement/precomputed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def precompute_vectors(name, vectors):
    """Precompute moves for a piece with static vectors."""
    print(f"Precomputing {name}...")
    
    # Create all board positions
    x, y, z = np.mgrid[0:SIZE, 0:SIZE, 0:SIZE]
    positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(COORD_DTYPE)
    
    # Result list (jagged array)
    all_moves = []
    
    for pos in positions:
        # Apply vectors
        targets = pos + vectors
        
        # Filter bounds
        valid_mask = in_bounds_vectorized(targets)
        valid_targets = targets[valid_mask]
        
        all_moves.append(valid_targets)
        
    # Save as object array (list of arrays)
    save_path = os.path.join(OUTPUT_DIR, f"moves_{name}.npy")
    np.save(save_path, np.array(all_moves, dtype=object))
    print(f"Saved {save_path}")

def precompute_edgerook():
    """Precompute moves for EdgeRook (BFS on edges)."""
    print("Precomputing EDGEROOK...")
    
    x, y, z = np.mgrid[0:SIZE, 0:SIZE, 0:SIZE]
    positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(COORD_DTYPE)
    
    all_moves = []
    
    for pos in positions:
        # Check if on edge
        if not (np.any(pos == 0) or np.any(pos == SIZE - 1)):
            all_moves.append(np.empty((0, 3), dtype=COORD_DTYPE))
            continue
            
        # BFS
        visited = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
        queue = [pos]
        visited[pos[0], pos[1], pos[2]] = True
        reachable = []
        
        idx = 0
        while idx < len(queue):
            curr = queue[idx]
            idx += 1
            
            cx, cy, cz = curr
            neighbors = _EDGE_NEIGHBORS[cx, cy, cz]
            
            for i in range(len(neighbors)):
                if neighbors[i, 0] == -1: continue # Invalid neighbor
                
                # Direction vector
                d = neighbors[i]
                nxt = curr + d
                
                nx, ny, nz = nxt
                if not visited[nx, ny, nz]:
                    visited[nx, ny, nz] = True
                    queue.append(nxt)
                    reachable.append(nxt)
                    
        if reachable:
            all_moves.append(np.array(reachable, dtype=COORD_DTYPE))
        else:
            all_moves.append(np.empty((0, 3), dtype=COORD_DTYPE))
            
    save_path = os.path.join(OUTPUT_DIR, "moves_EDGEROOK.npy")
    np.save(save_path, np.array(all_moves, dtype=object))
    print(f"Saved {save_path}")

def precompute_nebula():
    """Precompute moves for Nebula (Manhattan distance 3)."""
    print("Precomputing NEBULA...")
    
    # Reconstruct Nebula vectors
    directions = np.array([
        (-3, 0, 0), (3, 0, 0), (0, -3, 0), (0, 3, 0), (0, 0, -3), (0, 0, 3),
        (-2, -1, 0), (-2, 0, -1), (-2, 0, 1), (-2, 1, 0),
        (2, -1, 0), (2, 0, -1), (2, 0, 1), (2, 1, 0),
        (-1, -2, 0), (-1, 0, -2), (-1, 0, 2), (-1, 2, 0),
        (1, -2, 0), (1, 0, -2), (1, 0, 2), (1, 2, 0),
        (0, -2, -1), (0, -2, 1), (0, -1, -2), (0, -1, 2),
        (0, 1, -2), (0, 1, 2), (0, 2, -1), (0, 2, 1),
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ], dtype=COORD_DTYPE)
    
    precompute_vectors("NEBULA", directions)

def precompute_mirror():
    """Precompute moves for Mirror (Teleport)."""
    print("Precomputing MIRROR...")
    
    x, y, z = np.mgrid[0:SIZE, 0:SIZE, 0:SIZE]
    positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(COORD_DTYPE)
    
    all_moves = []
    
    for pos in positions:
        target = np.array([SIZE - 1 - pos[0], SIZE - 1 - pos[1], SIZE - 1 - pos[2]], dtype=COORD_DTYPE)
        
        if np.array_equal(pos, target):
            all_moves.append(np.empty((0, 3), dtype=COORD_DTYPE))
        else:
            all_moves.append(target.reshape(1, 3))
            
    save_path = os.path.join(OUTPUT_DIR, "moves_MIRROR.npy")
    np.save(save_path, np.array(all_moves, dtype=object))
    print(f"Saved {save_path}")

def precompute_archer():
    """Precompute moves for Archer (King + Radius 2)."""
    print("Precomputing ARCHER...")
    
    # King vectors
    king_vecs = KING_MOVEMENT_VECTORS
    
    # Archery vectors (Radius 2 surface)
    coords = np.mgrid[-2:3, -2:3, -2:3].reshape(3, -1).T
    distances = np.sum(coords * coords, axis=1)
    archery_vecs = coords[distances == 4].astype(COORD_DTYPE)
    
    combined = np.vstack([king_vecs, archery_vecs])
    precompute_vectors("ARCHER", combined)

def precompute_geomancer():
    """Precompute moves for Geomancer (King + Radius 2/3)."""
    print("Precomputing GEOMANCER...")
    
    king_vecs = KING_MOVEMENT_VECTORS
    
    # Radius 3 offsets
    offsets = np.asarray(RADIUS_3_OFFSETS, dtype=COORD_DTYPE)
    chebyshev_dist = np.max(np.abs(offsets), axis=1)
    geomancy_mask = chebyshev_dist >= 2
    geomancy_vecs = offsets[geomancy_mask]
    
    combined = np.vstack([king_vecs, geomancy_vecs])
    precompute_vectors("GEOMANCER", combined)

def precompute_bomb():
    """Precompute moves for Bomb (King + Self)."""
    print("Precomputing BOMB...")
    
    king_vecs = BOMB_MOVEMENT_VECTORS
    self_vec = np.array([[0, 0, 0]], dtype=COORD_DTYPE)
    
    combined = np.vstack([king_vecs, self_vec])
    precompute_vectors("BOMB", combined)

def main():
    # Standard vector pieces
    precompute_vectors("KNIGHT", KNIGHT_MOVEMENT_VECTORS)
    precompute_vectors("KING", KING_MOVEMENT_VECTORS)
    precompute_vectors("PRIEST", KING_MOVEMENT_VECTORS) # Same as King
    precompute_vectors("KNIGHT32", KNIGHT32_MOVEMENT_VECTORS)
    precompute_vectors("KNIGHT31", KNIGHT31_MOVEMENT_VECTORS)
    precompute_vectors("BLACKHOLE", BLACKHOLE_MOVEMENT_VECTORS)
    precompute_vectors("WHITEHOLE", KING_MOVEMENT_VECTORS) # King-like
    precompute_vectors("HIVE", HIVE_DIRECTIONS_3D)
    precompute_vectors("ORBITER", _ORBITAL_DIRS)
    precompute_vectors("ECHO", _ECHO_DIRECTIONS)
    precompute_vectors("PANEL", PANEL_MOVEMENT_VECTORS)
    precompute_vectors("FREEZER", FREEZER_MOVEMENT_VECTORS)
    precompute_vectors("WALL", WALL_MOVEMENT_VECTORS)
    precompute_vectors("ARMOUR", ARMOUR_MOVEMENT_VECTORS)
    precompute_vectors("SPEEDER", KING_MOVEMENT_VECTORS) # King-like
    precompute_vectors("SLOWER", SLOWER_MOVEMENT_VECTORS)
    precompute_vectors("SWAPPER", _SWAPPER_MOVEMENT_VECTORS) # Only static moves
    
    # Special pieces
    precompute_edgerook()
    precompute_nebula()
    precompute_mirror()
    precompute_archer()
    precompute_geomancer()
    precompute_bomb()

if __name__ == "__main__":
    main()
