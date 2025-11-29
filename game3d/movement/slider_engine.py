"""Slider Movement Engine - RAW MOVE GENERATION ONLY.

NO validation here. Just generates candidate moves.
Validation happens in generator.py.
"""

import numpy as np
from numba import njit, objmode
from typing import TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, INDEX_DTYPE, SIZE, SIZE_MINUS_1, SIZE_SQUARED
)
from game3d.movement.movepiece import Move

if TYPE_CHECKING:
    from game3d.cache.manager import UnifiedCacheManager


@njit(cache=True, fastmath=True)
def _generate_all_slider_moves(
    color: int,
    pos: np.ndarray,
    directions: np.ndarray,
    max_distance: int,
    flattened: np.ndarray,
    ignore_occupancy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-compiled slider move generation for all directions."""
    n_dirs = len(directions)
    max_possible = n_dirs * max_distance

    # Pre-allocate buffers
    moves = np.empty((max_possible, 3), dtype=COORD_DTYPE)
    captures = np.empty(max_possible, dtype=BOOL_DTYPE)
    write_idx = 0

    # Process each direction sequentially
    for d in range(n_dirs):
        direction = directions[d]
        
        # Skip zero vectors
        if direction[0] == 0 and direction[1] == 0 and direction[2] == 0:
            continue

        # Start from original position for each direction
        current_x = pos[0] + direction[0]
        current_y = pos[1] + direction[1]
        current_z = pos[2] + direction[2]

        for _ in range(max_distance):
            # Bounds check
            if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                break

            # Check occupancy
            idx = current_x + SIZE * current_y + SIZE_SQUARED * current_z
            occupant = flattened[idx]

            if occupant == 0:
                # Empty square - quiet move
                moves[write_idx, 0] = current_x
                moves[write_idx, 1] = current_y
                moves[write_idx, 2] = current_z
                captures[write_idx] = False
                write_idx += 1
            else:
                # Blocked
                if ignore_occupancy:
                    # Treat as a move (capture logic irrelevant for raw moves, but we mark it)
                    moves[write_idx, 0] = current_x
                    moves[write_idx, 1] = current_y
                    moves[write_idx, 2] = current_z
                    captures[write_idx] = (occupant != color) # Still mark capture if enemy
                    write_idx += 1
                    # CONTINUE RAY
                else:
                    if occupant != color:
                        # Capture move
                        moves[write_idx, 0] = current_x
                        moves[write_idx, 1] = current_y
                        moves[write_idx, 2] = current_z
                        captures[write_idx] = True
                        write_idx += 1
                    break  # Stop ray

            # Step forward
            current_x += direction[0]
            current_y += direction[1]
            current_z += direction[2]

    return moves[:write_idx], captures[:write_idx]


@njit(cache=True, fastmath=True)
def _generate_all_slider_moves_batch(
    color: int,
    positions: np.ndarray,
    directions: np.ndarray,
    max_distances: np.ndarray,
    flattened: np.ndarray,
    ignore_occupancy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-compiled slider move generation for BATCH of positions."""
    n_pos = positions.shape[0]
    n_dirs = directions.shape[0]
    
    # Estimate max moves (heuristic)
    # Use max possible distance from array
    global_max_dist = np.max(max_distances)
    max_moves_per_piece = n_dirs * global_max_dist
    total_max_moves = n_pos * max_moves_per_piece
    
    # Pre-allocate buffers
    moves = np.empty((total_max_moves, 6), dtype=COORD_DTYPE)
    
    write_idx = 0
    
    for i in range(n_pos):
        px, py, pz = positions[i]
        max_dist = max_distances[i]
        
        for d in range(n_dirs):
            dx, dy, dz = directions[d]
            
            # Skip zero vectors
            if dx == 0 and dy == 0 and dz == 0:
                continue
            
            current_x = px + dx
            current_y = py + dy
            current_z = pz + dz
            
            for _ in range(max_dist):
                # Bounds check
                if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                    break
                
                # Check occupancy
                idx = current_x + SIZE * current_y + SIZE_SQUARED * current_z
                occupant = flattened[idx]
                
                if occupant == 0:
                    # Empty square - quiet move
                    moves[write_idx, 0] = px
                    moves[write_idx, 1] = py
                    moves[write_idx, 2] = pz
                    moves[write_idx, 3] = current_x
                    moves[write_idx, 4] = current_y
                    moves[write_idx, 5] = current_z
                    write_idx += 1
                else:
                    # Blocked
                    if ignore_occupancy:
                        moves[write_idx, 0] = px
                        moves[write_idx, 1] = py
                        moves[write_idx, 2] = pz
                        moves[write_idx, 3] = current_x
                        moves[write_idx, 4] = current_y
                        moves[write_idx, 5] = current_z
                        write_idx += 1
                        # CONTINUE RAY
                    else:
                        if occupant != color:
                            # Capture move
                            moves[write_idx, 0] = px
                            moves[write_idx, 1] = py
                            moves[write_idx, 2] = pz
                            moves[write_idx, 3] = current_x
                            moves[write_idx, 4] = current_y
                            moves[write_idx, 5] = current_z
                            write_idx += 1
                        break  # Stop ray
                
                # Step forward
                current_x += dx
                current_y += dy
                current_z += dz
                
    return moves[:write_idx], np.empty(0, dtype=BOOL_DTYPE) # Return dummy captures



class SliderMovementEngine:
    """
    Raw slider move generation.

    NO validation - just generates candidate moves.
    Generator will validate them.
    """

    def __init__(self):
        pass  # No longer storing cache manager

    def generate_slider_moves_vectorized(
        self,
        cache_manager: 'UnifiedCacheManager',
        color: int,
        pos: np.ndarray,
        directions: np.ndarray,
        max_distance: int = SIZE_MINUS_1,
        ignore_occupancy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate slider moves using Numba-accelerated kernel."""
        pos_arr = np.asarray(pos, dtype=COORD_DTYPE).reshape(3)
        flattened = cache_manager.occupancy_cache.get_flattened_occupancy()

        return _generate_all_slider_moves(
            color, pos_arr, directions, max_distance, flattened, ignore_occupancy
        )

    def generate_slider_moves(
        self,
        cache_manager: 'UnifiedCacheManager',
        color: int,
        pos: np.ndarray,
        directions: np.ndarray,
        max_distance: int = SIZE_MINUS_1,
        ignore_occupancy: bool = False
    ) -> list[Move]:
        """Generate slider moves as Move objects (NO validation)."""
        moves, captures = self.generate_slider_moves_vectorized(
            cache_manager, color, pos, directions, max_distance, ignore_occupancy
        )

        # Create Move objects
        return Move.create_batch(pos, moves, captures)

    def generate_slider_moves_array(
        self,
        cache_manager: 'UnifiedCacheManager',
        color: int,
        pos: np.ndarray,
        directions: np.ndarray,
        max_distance: int = SIZE_MINUS_1,
        ignore_occupancy: bool = False
    ) -> np.ndarray:
        """Generate slider moves as numpy array [from_x, from_y, from_z, to_x, to_y, to_z]."""
        # Check for batch input
        if pos.ndim == 2:
            flattened = cache_manager.occupancy_cache.get_flattened_occupancy()
            
            # Ensure max_distances is an array
            if isinstance(max_distance, (int, np.integer)):
                max_dists = np.full(pos.shape[0], max_distance, dtype=np.int32)
            else:
                max_dists = max_distance.astype(np.int32)
                
            moves, _ = _generate_all_slider_moves_batch(
                color, pos, directions, max_dists, flattened, ignore_occupancy
            )
            return moves

        # Legacy single position path
        destinations, captures = self.generate_slider_moves_vectorized(
            cache_manager, color, pos, directions, max_distance, ignore_occupancy
        )
        
        if destinations.size == 0:
            return np.empty((0, 6), dtype=COORD_DTYPE)
            
        n_moves = destinations.shape[0]
        moves = np.empty((n_moves, 6), dtype=COORD_DTYPE)
        
        # Fill from_coord
        moves[:, 0] = pos[0]
        moves[:, 1] = pos[1]
        moves[:, 2] = pos[2]
        
        # Fill to_coord
        moves[:, 3:6] = destinations
        
        return moves


def get_slider_movement_generator() -> SliderMovementEngine:
    """Factory function to create a SliderMovementEngine instance."""
    return SliderMovementEngine()


__all__ = ['SliderMovementEngine', 'get_slider_movement_generator']
