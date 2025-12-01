# edgerook.py - OPTIMIZED BATCH NUMBA VERSION
"""Edge-Rook piece - fully numpy native edge graph traversal."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from numba import njit, prange

from game3d.common.shared_types import SIZE, SIZE_MINUS_1, COORD_DTYPE, BOOL_DTYPE
from game3d.common.shared_types import Color, PieceType
from game3d.common.registry import register
from game3d.movement.jump_engine import get_jump_movement_generator
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.game.gamestate import GameState
    from game3d.cache.manager import OptimizedCacheManager

# Edge-Rook movement vectors - 6 orthogonal directions
EDGE_ROOK_VECTORS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=COORD_DTYPE)

N_EDGE_VECTORS = len(EDGE_ROOK_VECTORS)

# Precomputed edge adjacency cache
_EDGE_NEIGHBORS: np.ndarray = None

def _build_edge_graph() -> np.ndarray:
    """Build edge adjacency matrix using vectorized numpy operations."""
    neighbors = -np.ones((SIZE, SIZE, SIZE, N_EDGE_VECTORS, 3), dtype=COORD_DTYPE)

    # Create coordinate grids
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(SIZE, dtype=COORD_DTYPE),
        np.arange(SIZE, dtype=COORD_DTYPE),
        np.arange(SIZE, dtype=COORD_DTYPE),
        indexing='ij'
    )

    # Identify edge cells
    edge_mask = (
        (x_grid == 0) | (x_grid == SIZE_MINUS_1) |
        (y_grid == 0) | (y_grid == SIZE_MINUS_1) |
        (z_grid == 0) | (z_grid == SIZE_MINUS_1)
    )

    # Process each direction
    for dir_idx, direction in enumerate(EDGE_ROOK_VECTORS):
        dx, dy, dz = direction

        # Calculate neighbor coordinates
        nx_grid = x_grid + dx
        ny_grid = y_grid + dy
        nz_grid = z_grid + dz

        # Check bounds
        in_bounds = (
            (nx_grid >= 0) & (nx_grid < SIZE) &
            (ny_grid >= 0) & (ny_grid < SIZE) &
            (nz_grid >= 0) & (nz_grid < SIZE)
        )

        # Check if neighbor is also edge cell
        edge_neighbors = (
            (nx_grid == 0) | (nx_grid == SIZE_MINUS_1) |
            (ny_grid == 0) | (ny_grid == SIZE_MINUS_1) |
            (nz_grid == 0) | (nz_grid == SIZE_MINUS_1)
        )

        # Valid edge neighbors
        valid = in_bounds & edge_neighbors & edge_mask

        # Assign direction vectors
        neighbors[valid, dir_idx, :] = direction

    return neighbors

# Initialize adjacency cache
_EDGE_NEIGHBORS = _build_edge_graph()

@njit(cache=True, fastmath=True)
def _edgerook_bfs_batch_numba(
    start_nodes: np.ndarray,
    neighbors_table: np.ndarray,
    flattened_occ: np.ndarray,
    color: int
) -> np.ndarray:
    """
    Numba-optimized BFS for Edge Rook batch.
    Performs BFS for each start node.
    """
    n_starts = start_nodes.shape[0]
    
    # Max moves per piece is small (edge graph is sparse)
    # But to be safe, let's allocate a decent buffer
    max_moves_per_piece = SIZE * SIZE * 6
    total_max_moves = n_starts * max_moves_per_piece
    
    # Output buffer
    all_moves = np.empty((total_max_moves, 6), dtype=COORD_DTYPE)
    total_count = 0
    
    # Reusable buffers for BFS (per piece)
    queue = np.empty((max_moves_per_piece, 3), dtype=COORD_DTYPE)
    
    for i in range(n_starts):
        start_node = start_nodes[i]
        start_x, start_y, start_z = start_node
        
        # Reset BFS state
        q_read = 0
        q_write = 0
        
        # Visited array (reset for each piece)
        # Optimization: Could use a generation counter to avoid clearing, 
        # but for now simple reset is safer and fast enough for small 9x9x9
        visited = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
        
        # Initialize BFS
        queue[q_write] = start_node
        q_write += 1
        visited[start_x, start_y, start_z] = True
        
        while q_read < q_write:
            # Pop
            curr = queue[q_read]
            q_read += 1
            cx, cy, cz = curr[0], curr[1], curr[2]
            
            for j in range(6):
                # Direct lookup
                dx = neighbors_table[cx, cy, cz, j, 0]
                if dx == -1: continue # Invalid neighbor
                
                dy = neighbors_table[cx, cy, cz, j, 1]
                dz = neighbors_table[cx, cy, cz, j, 2]
                
                tx, ty, tz = cx + dx, cy + dy, cz + dz
                
                if not visited[tx, ty, tz]:
                    visited[tx, ty, tz] = True
                    
                    # Check occupancy
                    flat_idx = tx + SIZE * ty + SIZE * SIZE * tz
                    occ_val = flattened_occ[flat_idx]
                    
                    if occ_val == 0:
                        # Empty: Add to queue and record move
                        queue[q_write, 0] = tx
                        queue[q_write, 1] = ty
                        queue[q_write, 2] = tz
                        q_write += 1
                        
                        all_moves[total_count, 0] = start_x
                        all_moves[total_count, 1] = start_y
                        all_moves[total_count, 2] = start_z
                        all_moves[total_count, 3] = tx
                        all_moves[total_count, 4] = ty
                        all_moves[total_count, 5] = tz
                        total_count += 1
                        
                    elif occ_val != color:
                        # Enemy: Capture and Stop BFS (don't add to queue)
                        all_moves[total_count, 0] = start_x
                        all_moves[total_count, 1] = start_y
                        all_moves[total_count, 2] = start_z
                        all_moves[total_count, 3] = tx
                        all_moves[total_count, 4] = ty
                        all_moves[total_count, 5] = tz
                        total_count += 1
                        
    return all_moves[:total_count]

def generate_edgerook_moves(cache_manager, color: int, pos: np.ndarray) -> np.ndarray:
    """Optimized wrapper using Numba implementation."""
    pos_arr = pos.astype(COORD_DTYPE)
    
    # Handle single input
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)
        
    if pos_arr.shape[0] == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Fast exit check - filter pieces not on edge?
    # Actually, if a piece is not on edge, it has no neighbors in the graph, 
    # so BFS will just terminate immediately. No need for explicit filter.

    # Get cached flattened occupancy (O(1) access)
    flattened = cache_manager.occupancy_cache.get_cached_flattened()

    # Run Numba BFS
    return _edgerook_bfs_batch_numba(pos_arr, _EDGE_NEIGHBORS, flattened, color)

@register(PieceType.EDGEROOK)
def edgerook_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatcher for edge-rook moves."""
    return generate_edgerook_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_edgerook_moves", "EDGE_ROOK_VECTORS"]
