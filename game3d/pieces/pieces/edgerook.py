"""Edge-Rook piece - fully numpy native edge graph traversal."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

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

def generate_edgerook_moves(
    cache_manager: 'OptimizedCacheManager',
    color: int,
    pos: np.ndarray
) -> np.ndarray:
    """Generate edge-rook moves using numpy-native BFS."""
    start = pos.astype(COORD_DTYPE).ravel()

    # Early exit if not on edge
    if not np.any((start == 0) | (start == SIZE_MINUS_1)):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # BFS setup
    visited = np.zeros((SIZE, SIZE, SIZE), dtype=BOOL_DTYPE)
    queue = start.reshape(1, 3)
    visited[int(start[2]), int(start[1]), int(start[0])] = True

    flattened = cache_manager.occupancy_cache.get_flattened_occupancy()
    reachable_targets = []

    # BFS traversal
    while queue.shape[0] > 0:
        current = queue[0]
        queue = queue[1:]

        cx, cy, cz = current.astype(int)
        neighbors = _EDGE_NEIGHBORS[cx, cy, cz]

        # Filter valid neighbors
        valid_mask = neighbors[:, 0] != -1
        if not np.any(valid_mask):
            continue

        valid_neighbors = neighbors[valid_mask]
        target_coords = current + valid_neighbors
        target_indices = target_coords.astype(int)

        # Check visited status
        unvisited_mask = ~visited[target_indices[:, 2], target_indices[:, 1], target_indices[:, 0]]
        if not np.any(unvisited_mask):
            continue

        unvisited_targets = target_coords[unvisited_mask]
        unvisited_indices = target_indices[unvisited_mask]

        # Mark as visited
        visited[unvisited_indices[:, 2], unvisited_indices[:, 1], unvisited_indices[:, 0]] = True

        # Check occupancy
        flat_idxs = (unvisited_targets[:, 0] + SIZE * unvisited_targets[:, 1] + SIZE * SIZE * unvisited_targets[:, 2]).astype(int)
        occs = flattened[flat_idxs]

        # Empty cells continue BFS
        empty_mask = occs == 0
        if np.any(empty_mask):
            empty_targets = unvisited_targets[empty_mask]
            queue = np.vstack([queue, empty_targets]) if queue.size > 0 else empty_targets

        # All unvisited targets are reachable
        reachable_targets.append(unvisited_targets)

    if not reachable_targets:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Combine targets
    targets = np.vstack(reachable_targets)

    # Remove start position
    distances = np.sum((targets - start) ** 2, axis=1)
    mask = distances > 0
    targets = targets[mask]

    if targets.size == 0:
        return np.empty((0, 6), dtype=COORD_DTYPE)

    # Filter valid moves and determine captures
    flat_idxs = (targets[:, 0] + SIZE * targets[:, 1] + SIZE * SIZE * targets[:, 2]).astype(int)
    occs = flattened[flat_idxs]

    valid_mask = occs != color
    if not np.any(valid_mask):
        return np.empty((0, 6), dtype=COORD_DTYPE)

    final_targets = targets[valid_mask]
    
    # Create move array: [from_x, from_y, from_z, to_x, to_y, to_z]
    n_moves = final_targets.shape[0]
    move_array = np.empty((n_moves, 6), dtype=COORD_DTYPE)
    move_array[:, 0:3] = start
    move_array[:, 3:6] = final_targets
    
    return move_array

@register(PieceType.EDGEROOK)
def edgerook_move_dispatcher(state: 'GameState', pos: np.ndarray) -> np.ndarray:
    """Dispatcher for edge-rook moves."""
    return generate_edgerook_moves(state.cache_manager, state.color, pos)

__all__ = ["generate_edgerook_moves", "EDGE_ROOK_VECTORS"]
