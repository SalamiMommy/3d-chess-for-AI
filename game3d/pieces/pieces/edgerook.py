"""Edge-Rook piece - fully numpy native edge graph traversal."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from numba import njit

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
def _edgerook_bfs_numba(start_node: np.ndarray,
                        neighbors_table: np.ndarray,
                        flattened_occ: np.ndarray,
                        color: int) -> np.ndarray:
    """
    Numba-optimized BFS for Edge Rook.
    Eliminates Python loop overhead and np.vstack memory reallocation.
    """
    # 1. Pre-allocate max possible queue (surface area of cube is small enough)
    max_queue = SIZE * SIZE * 6
    queue = np.empty((max_queue, 3), dtype=COORD_DTYPE)
    q_read = 0
    q_write = 0

    # 2. Visited array
    visited = np.zeros((SIZE, SIZE, SIZE), dtype=np.bool_)

    # 3. Output buffer (max possible moves)
    found_targets = np.empty((max_queue, 3), dtype=COORD_DTYPE)
    found_count = 0

    # Initialize
    start_x, start_y, start_z = start_node
    queue[q_write] = start_node
    q_write += 1
    visited[start_x, start_y, start_z] = True

    while q_read < q_write:
        # Pop
        curr = queue[q_read]
        q_read += 1
        cx, cy, cz = curr[0], curr[1], curr[2]

        # Get neighbors from precomputed table
        # neighbors_table shape: (SIZE, SIZE, SIZE, 6, 3)
        # We assume invalid neighbors are marked with -1

        for i in range(6):
            # Direct lookup is faster than slicing in Numba
            dx = neighbors_table[cx, cy, cz, i, 0]
            if dx == -1: continue # Invalid neighbor (cached as -1)

            dy = neighbors_table[cx, cy, cz, i, 1]
            dz = neighbors_table[cx, cy, cz, i, 2]

            # Target Coordinate
            tx, ty, tz = cx + dx, cy + dy, cz + dz

            if not visited[tx, ty, tz]:
                visited[tx, ty, tz] = True

                # Check occupancy (linear index)
                flat_idx = tx + SIZE * ty + SIZE * SIZE * tz
                occ_val = flattened_occ[flat_idx]

                if occ_val == 0:
                    # Empty: Add to queue and Continue BFS
                    queue[q_write, 0] = tx
                    queue[q_write, 1] = ty
                    queue[q_write, 2] = tz
                    q_write += 1

                    # Add to valid targets
                    found_targets[found_count, 0] = tx
                    found_targets[found_count, 1] = ty
                    found_targets[found_count, 2] = tz
                    found_count += 1

                elif occ_val != color:
                    # Enemy: Capture and Stop BFS (don't add to queue)
                    found_targets[found_count, 0] = tx
                    found_targets[found_count, 1] = ty
                    found_targets[found_count, 2] = tz
                    found_count += 1
                    # Blocked by piece, do not queue

                # If friendly (occ_val == color), do nothing (blocked)

    # Format result: (N, 6)
    if found_count == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    result = np.empty((found_count, 6), dtype=COORD_DTYPE)
    # Broadcast start position
    result[:, 0] = start_x
    result[:, 1] = start_y
    result[:, 2] = start_z
    # Fill targets
    result[:, 3:6] = found_targets[:found_count]

    return result

def generate_edgerook_moves(cache_manager, color: int, pos: np.ndarray) -> np.ndarray:
    """Optimized wrapper using Numba implementation."""
    # Fast exit check
    if not (np.any(pos == 0) or np.any(pos == SIZE_MINUS_1)):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Get cached flattened occupancy (O(1) access)
    flattened = cache_manager.occupancy_cache.get_cached_flattened()

    # Run Numba BFS
    return _edgerook_bfs_numba(pos.astype(COORD_DTYPE), _EDGE_NEIGHBORS, flattened, color)

@register(PieceType.EDGEROOK)
def edgerook_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatcher for edge-rook moves."""
    return generate_edgerook_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_edgerook_moves", "EDGE_ROOK_VECTORS"]
