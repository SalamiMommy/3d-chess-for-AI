"""Slider Movement Engine - RAW MOVE GENERATION ONLY.

NO validation here. Just generates candidate moves.
Validation happens in generator.py.
"""

import numpy as np
from numba import njit, objmode, prange
from typing import TYPE_CHECKING

from game3d.common.shared_types import (
    COORD_DTYPE, BOOL_DTYPE, INDEX_DTYPE, SIZE, SIZE_MINUS_1, SIZE_SQUARED
)
from game3d.movement.movepiece import Move
import os

if TYPE_CHECKING:
    from game3d.cache.manager import UnifiedCacheManager


# Global cache for precomputed data
# Key: piece_name, Value: (rays_flat, ray_offsets, square_offsets)
_PRECOMPUTED_CACHE = {}

def _load_precomputed_data(piece_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed ray data for a piece type."""
    if piece_name in _PRECOMPUTED_CACHE:
        return _PRECOMPUTED_CACHE[piece_name]
    
    base_dir = os.path.join(os.path.dirname(__file__), "precomputed")
    flat_path = os.path.join(base_dir, f"rays_{piece_name}_flat.npy")
    ray_offsets_path = os.path.join(base_dir, f"rays_{piece_name}_ray_offsets.npy")
    sq_offsets_path = os.path.join(base_dir, f"rays_{piece_name}_sq_offsets.npy")
    
    if not os.path.exists(flat_path):
        # Fallback or error? For now error to ensure we know if something is missing
        raise FileNotFoundError(f"Precomputed data for {piece_name} not found in {base_dir}")
        
    rays_flat = np.load(flat_path)
    ray_offsets = np.load(ray_offsets_path)
    sq_offsets = np.load(sq_offsets_path)
    
    data = (rays_flat, ray_offsets, sq_offsets)
    _PRECOMPUTED_CACHE[piece_name] = data
    return data


@njit(cache=True, fastmath=True)
def _generate_all_slider_moves(
    color: int,
    pos: np.ndarray,
    directions: np.ndarray,
    max_distance: int,
    occ: np.ndarray,
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

        # DEBUG
        if direction[0] == 1 and direction[1] == 2 and direction[2] == 0:
            print("Checking dir [1, 2, 0]")
            print("Start:", current_x, current_y, current_z)

        for _ in range(max_distance):
            # Bounds check
            if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                break

            # Check occupancy
            occupant = occ[current_x, current_y, current_z]

            if occupant == 0:
                # Empty square - quiet move
                moves[write_idx, 0] = current_x
                moves[write_idx, 1] = current_y
                moves[write_idx, 2] = current_z
                captures[write_idx] = False
                write_idx += 1
            else:
                # DEBUG
                if direction[0] == 1 and direction[1] == 2 and direction[2] == 0:
                    print("Hit occupant:", occupant, "at", current_x, current_y, current_z)
                    print("Color:", color, "Ignore:", ignore_occupancy)

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


@njit(cache=True, fastmath=True, parallel=True)
def _generate_all_slider_moves_batch(
    color: int,
    positions: np.ndarray,
    directions: np.ndarray,
    max_distances: np.ndarray,
    occ: np.ndarray,
    ignore_occupancy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-compiled slider move generation for BATCH of positions."""
    n_pos = positions.shape[0]
    n_dirs = directions.shape[0]
    
    # Pass 1: Count moves per piece
    counts = np.zeros(n_pos, dtype=np.int32)
    
    for i in prange(n_pos):
        px, py, pz = positions[i]
        max_dist = max_distances[i]
        count = 0
        
        for d in range(n_dirs):
            dx, dy, dz = directions[d]
            
            if dx == 0 and dy == 0 and dz == 0:
                continue
                
            current_x = px + dx
            current_y = py + dy
            current_z = pz + dz
            
            for _ in range(max_dist):
                if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                    break
                
                occupant = occ[current_x, current_y, current_z]
                
                if occupant == 0:
                    count += 1
                else:
                    if ignore_occupancy:
                        count += 1
                    else:
                        if occupant != color:
                            count += 1
                        break
                
                current_x += dx
                current_y += dy
                current_z += dz
        
        counts[i] = count
        
        # DEBUG PRINT
        # if i == 0:
        #     print(f"Slider Batch Debug: pos={positions[i]}, count={count}")
        
    # Pass 2: Calculate offsets
    total_moves = np.sum(counts)
    offsets = np.zeros(n_pos, dtype=np.int32)
    current_offset = 0
    for i in range(n_pos):
        offsets[i] = current_offset
        current_offset += counts[i]
        
    # Pass 3: Fill moves
    moves = np.empty((total_moves, 6), dtype=COORD_DTYPE)
    
    for i in prange(n_pos):
        write_idx = offsets[i]
        px, py, pz = positions[i]
        max_dist = max_distances[i]
        
        for d in range(n_dirs):
            dx, dy, dz = directions[d]
            
            if dx == 0 and dy == 0 and dz == 0:
                continue
                
            current_x = px + dx
            current_y = py + dy
            current_z = pz + dz
            
            for _ in range(max_dist):
                if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                    break
                
                occupant = occ[current_x, current_y, current_z]
                
                should_write = False
                if occupant == 0:
                    should_write = True
                else:
                    if ignore_occupancy:
                        should_write = True
                    else:
                        if occupant != color:
                            should_write = True
                        # Break comes after writing
                
                if should_write:
                    moves[write_idx, 0] = px
                    moves[write_idx, 1] = py
                    moves[write_idx, 2] = pz
                    moves[write_idx, 3] = current_x
                    moves[write_idx, 4] = current_y
                    moves[write_idx, 5] = current_z
                    write_idx += 1
                
                if occupant != 0:
                    if not ignore_occupancy:
                        break
                
                current_x += dx
                current_y += dy
                current_z += dz
                
    return moves, np.empty(0, dtype=BOOL_DTYPE) # Return dummy captures


@njit(cache=True, fastmath=True, parallel=False)
def _generate_all_slider_moves_batch_serial(
    color: int,
    positions: np.ndarray,
    directions: np.ndarray,
    max_distances: np.ndarray,
    occ: np.ndarray,
    ignore_occupancy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Serial version of slider move generation for BATCH of positions."""
    n_pos = positions.shape[0]
    n_dirs = directions.shape[0]
    
    # Pass 1: Count moves per piece
    counts = np.zeros(n_pos, dtype=np.int32)
    
    for i in range(n_pos):
        px, py, pz = positions[i]
        max_dist = max_distances[i]
        count = 0
        
        for d in range(n_dirs):
            dx, dy, dz = directions[d]
            
            if dx == 0 and dy == 0 and dz == 0:
                continue
                
            current_x = px + dx
            current_y = py + dy
            current_z = pz + dz
            
            for _ in range(max_dist):
                if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                    break
                
                occupant = occ[current_x, current_y, current_z]
                
                if occupant == 0:
                    count += 1
                else:
                    if ignore_occupancy:
                        count += 1
                    else:
                        if occupant != color:
                            count += 1
                        break
                
                current_x += dx
                current_y += dy
                current_z += dz
        
        counts[i] = count
        
    # Pass 2: Calculate offsets
    total_moves = np.sum(counts)
    offsets = np.zeros(n_pos, dtype=np.int32)
    current_offset = 0
    for i in range(n_pos):
        offsets[i] = current_offset
        current_offset += counts[i]
        
    # Pass 3: Fill moves
    moves = np.empty((total_moves, 6), dtype=COORD_DTYPE)
    
    for i in range(n_pos):
        write_idx = offsets[i]
        px, py, pz = positions[i]
        max_dist = max_distances[i]
        
        for d in range(n_dirs):
            dx, dy, dz = directions[d]
            
            if dx == 0 and dy == 0 and dz == 0:
                continue
                
            current_x = px + dx
            current_y = py + dy
            current_z = pz + dz
            
            for _ in range(max_dist):
                if not (0 <= current_x < SIZE and 0 <= current_y < SIZE and 0 <= current_z < SIZE):
                    break
                
                occupant = occ[current_x, current_y, current_z]
                
                should_write = False
                if occupant == 0:
                    should_write = True
                else:
                    if ignore_occupancy:
                        should_write = True
                    else:
                        if occupant != color:
                            should_write = True
                        # Break comes after writing
                
                if should_write:
                    moves[write_idx, 0] = px
                    moves[write_idx, 1] = py
                    moves[write_idx, 2] = pz
                    moves[write_idx, 3] = current_x
                    moves[write_idx, 4] = current_y
                    moves[write_idx, 5] = current_z
                    write_idx += 1
                
                if occupant != 0:
                    if not ignore_occupancy:
                        break
                
                current_x += dx
                current_y += dy
                current_z += dz
                
    return moves, np.empty(0, dtype=BOOL_DTYPE) # Return dummy captures


    return moves, np.empty(0, dtype=BOOL_DTYPE) # Return dummy captures


@njit(cache=True, fastmath=True)
def _generate_precomputed_slider_moves(
    color: int,
    pos_flat_idx: int,
    rays_flat: np.ndarray,
    ray_offsets: np.ndarray,
    square_offsets: np.ndarray,
    occ: np.ndarray,
    ignore_occupancy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate slider moves using precomputed rays.
    
    rays_flat: (N, 3) array of all ray coordinates
    ray_offsets: (M+1) array of indices into rays_flat
    square_offsets: (SIZE^3 + 1) array of indices into ray_offsets
    """
    
    # Get range of rays for this square
    start_ray_idx = square_offsets[pos_flat_idx]
    end_ray_idx = square_offsets[pos_flat_idx + 1]
    
    # Pass 1: Count valid moves to allocate arrays
    count = 0
    
    for r_idx in range(start_ray_idx, end_ray_idx):
        start_coord_idx = ray_offsets[r_idx]
        end_coord_idx = ray_offsets[r_idx + 1]
        
        for c_idx in range(start_coord_idx, end_coord_idx):
            cx = rays_flat[c_idx, 0]
            cy = rays_flat[c_idx, 1]
            cz = rays_flat[c_idx, 2]
            
            occupant = occ[cx, cy, cz]
            
            if occupant == 0:
                count += 1
            else:
                if ignore_occupancy:
                    count += 1
                else:
                    if occupant != color:
                        count += 1
                    break # Stop ray
    
    # Allocate
    moves = np.empty((count, 3), dtype=COORD_DTYPE)
    captures = np.empty(count, dtype=BOOL_DTYPE)
    write_idx = 0
    
    # Pass 2: Fill moves
    for r_idx in range(start_ray_idx, end_ray_idx):
        start_coord_idx = ray_offsets[r_idx]
        end_coord_idx = ray_offsets[r_idx + 1]
        
        for c_idx in range(start_coord_idx, end_coord_idx):
            cx = rays_flat[c_idx, 0]
            cy = rays_flat[c_idx, 1]
            cz = rays_flat[c_idx, 2]
            
            occupant = occ[cx, cy, cz]
            
            should_write = False
            stop_ray = False
            
            if occupant == 0:
                should_write = True
            else:
                if ignore_occupancy:
                    should_write = True
                else:
                    if occupant != color:
                        should_write = True
                    stop_ray = True
            
            if should_write:
                moves[write_idx, 0] = cx
                moves[write_idx, 1] = cy
                moves[write_idx, 2] = cz
                if occupant != 0 and occupant != color:
                    captures[write_idx] = True
                else:
                    captures[write_idx] = False
                write_idx += 1
            
            if stop_ray:
                break
                
    return moves, captures


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
        occ = cache_manager.occupancy_cache._occ

        return _generate_all_slider_moves(
            color, pos_arr, directions, max_distance, occ, ignore_occupancy
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
            occ = cache_manager.occupancy_cache._occ
            
            # Ensure max_distances is an array
            if isinstance(max_distance, (int, np.integer)):
                max_dists = np.full(pos.shape[0], max_distance, dtype=np.int32)
            else:
                max_dists = max_distance.astype(np.int32)
            
            # Threshold for parallel execution
            # Based on benchmark, serial is faster for < 250 items
            if pos.shape[0] < 250:
                moves, _ = _generate_all_slider_moves_batch_serial(
                    color, pos, directions, max_dists, occ, ignore_occupancy
                )
            else:
                moves, _ = _generate_all_slider_moves_batch(
                    color, pos, directions, max_dists, occ, ignore_occupancy
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


    def generate_slider_moves_precomputed(
        self,
        cache_manager: 'UnifiedCacheManager',
        color: int,
        pos: np.ndarray,
        piece_name: str,
        ignore_occupancy: bool = False
    ) -> np.ndarray:
        """
        Generate slider moves using precomputed rays for the given piece type.
        Returns array of [from_x, from_y, from_z, to_x, to_y, to_z].
        """
        # Ensure data is loaded
        rays_flat, ray_offsets, sq_offsets = _load_precomputed_data(piece_name)
        
        occ = cache_manager.occupancy_cache._occ
        
        # Handle batch input
        if pos.ndim == 2:
            move_lists = []
            
            for i in range(pos.shape[0]):
                p = pos[i]
                flat_idx = p[0] + p[1] * SIZE + p[2] * SIZE_SQUARED
                
                dests, _ = _generate_precomputed_slider_moves(
                    color, flat_idx, rays_flat, ray_offsets, sq_offsets, occ, ignore_occupancy
                )
                
                if dests.size > 0:
                    n = dests.shape[0]
                    m = np.empty((n, 6), dtype=COORD_DTYPE)
                    m[:, 0:3] = p
                    m[:, 3:6] = dests
                    move_lists.append(m)
            
            if not move_lists:
                return np.empty((0, 6), dtype=COORD_DTYPE)
            return np.concatenate(move_lists)

        # Single position
        pos_arr = pos.astype(COORD_DTYPE).reshape(3)
        flat_idx = pos_arr[0] + pos_arr[1] * SIZE + pos_arr[2] * SIZE_SQUARED
        
        dests, _ = _generate_precomputed_slider_moves(
            color, flat_idx, rays_flat, ray_offsets, sq_offsets, occ, ignore_occupancy
        )
        
        if dests.size == 0:
            return np.empty((0, 6), dtype=COORD_DTYPE)
            
        n_moves = dests.shape[0]
        moves = np.empty((n_moves, 6), dtype=COORD_DTYPE)
        
        moves[:, 0] = pos_arr[0]
        moves[:, 1] = pos_arr[1]
        moves[:, 2] = pos_arr[2]
        moves[:, 3:6] = dests
        
        return moves


def get_slider_movement_generator() -> SliderMovementEngine:
    """Factory function to create a SliderMovementEngine instance."""
    return SliderMovementEngine()


__all__ = ['SliderMovementEngine', 'get_slider_movement_generator']
