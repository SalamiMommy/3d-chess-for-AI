# game3d/board/board.py
"""Fully optimized numpy-native board with vectorized operations."""

import numpy as np
from typing import Optional, Union, Tuple
from numba import njit, prange
import logging
logger = logging.getLogger(__name__)

from game3d.common.shared_types import (
    SIZE, N_TOTAL_PLANES, COORD_DTYPE, FLOAT_DTYPE, INDEX_DTYPE, VECTORIZATION_THRESHOLD,
    COLOR_WHITE, COLOR_BLACK, MAX_COORD_VALUE, N_PIECE_TYPES, PieceType, Color, COLOR_DTYPE,
    PIECE_TYPE_DTYPE
)
from game3d.common.coord_utils import in_bounds_vectorized
from game3d.common.validation import validate_coords_batch as validate_coords

# =============================================================================
# VECTORIZED BOARD UTILITIES
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def _batch_get_pieces_at_coords(board: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Get pieces at multiple coordinates - fully vectorized."""
    n_coords = coords.shape[0]
    results = np.empty(n_coords, dtype=FLOAT_DTYPE)

    for i in prange(n_coords):
        x, y, z = coords[i]
        if 0 <= x <= MAX_COORD_VALUE and 0 <= y <= MAX_COORD_VALUE and 0 <= z <= MAX_COORD_VALUE:
            results[i] = board[:, x, y, z].max()
        else:
            results[i] = 0.0

    return results

@njit(cache=True, fastmath=True, parallel=True)
def _batch_set_pieces_at_coords(board: np.ndarray, coords: np.ndarray,
                              piece_values: np.ndarray) -> None:
    """Set pieces at multiple coordinates - fully vectorized."""
    n_coords = coords.shape[0]

    for i in prange(n_coords):
        x, y, z = coords[i]
        if 0 <= x <= MAX_COORD_VALUE and 0 <= y <= MAX_COORD_VALUE and 0 <= z <= MAX_COORD_VALUE:
            board[:, x, y, z] = 0.0
            if piece_values[i] > 0:
                board[0, x, y, z] = piece_values[i]

# =============================================================================
# OPTIMIZED BOARD CLASS
# =============================================================================
class Board:
    """
    Stateless board configuration factory.
    
    This class NO LONGER holds the game state. It serves only to:
    1. Define the initial starting position (coordinates, types, colors).
    2. Provide static configuration constants if needed.
    
    The actual game state lives exclusively in the OccupancyCache.
    """
    __slots__ = ('_cache_manager', 'generation')

    def __init__(self):
        """Initialize board configuration."""
        self._cache_manager = None
        self.generation = 0  # Kept for compatibility, but should not be relied upon for state

    @property
    def cache_manager(self):
        """Get the cache manager associated with this board."""
        return self._cache_manager

    @cache_manager.setter
    def cache_manager(self, value):
        """Set the cache manager for this board."""
        self._cache_manager = value

    def array(self) -> np.ndarray:
        """
        DEPRECATED: Get underlying board array.
        
        WARNING: This method now returns a reconstructed array from the cache if available,
        or throws an error if no cache manager is attached.
        """
        if self._cache_manager:
            # Reconstruct array from cache for backward compatibility
            # This is expensive and should be avoided in hot paths
            return self._reconstruct_array_from_cache()
        
        # Fallback for initialization before cache is attached (should be empty)
        return np.zeros((N_TOTAL_PLANES, SIZE, SIZE, SIZE), dtype=FLOAT_DTYPE)

    def get_board_array(self) -> np.ndarray:
        """Get board array (alias for array())."""
        return self.array()

    def _reconstruct_array_from_cache(self) -> np.ndarray:
        """Reconstruct the full 4D board array from the OccupancyCache."""
        arr = np.zeros((N_TOTAL_PLANES, SIZE, SIZE, SIZE), dtype=FLOAT_DTYPE)
        
        coords, types, colors = self._cache_manager.occupancy_cache.get_all_occupied_vectorized()
        
        if coords.shape[0] == 0:
            return arr
            
        # Vectorized reconstruction
        color_offsets = (colors - Color.WHITE).astype(INDEX_DTYPE)
        plane_indices = (types - 1) + (color_offsets * N_PIECE_TYPES)
        
        x = coords[:, 0].astype(INDEX_DTYPE)
        y = coords[:, 1].astype(INDEX_DTYPE)
        z = coords[:, 2].astype(INDEX_DTYPE)
        
        arr[plane_indices, x, y, z] = 1.0
        return arr

    @staticmethod
    def empty() -> 'Board':
        """Create empty board configuration."""
        return Board()

    @classmethod
    def startpos(cls) -> 'Board':
        """Create board configuration for starting position."""
        return cls()

    def get_initial_setup(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the initial starting position configuration.
        
        Returns:
            Tuple containing (coords, types, colors) for the starting position.
        """
        # Define layouts as direct integer arrays (PieceType enum values)
        # Rank 1 layout (back rank) - 9x9 grid of piece types
        rank1_layout = np.array([
            [35, 21, 16, 14, 12, 14, 16, 21, 35],  # Top row: reflector, coneslider, etc.
            [40, 33, 18, 19, 22, 19, 18, 33, 40],
            [34, 27, 15, 11, 9, 11, 15, 27, 34],
            [26, 32, 13, 8, 39, 8, 13, 32, 26],
            [12, 22, 9, 39, 6, 39, 9, 22, 12],
            [26, 32, 13, 8, 39, 8, 13, 32, 26],
            [34, 27, 15, 11, 9, 11, 15, 27, 34],
            [40, 33, 18, 19, 22, 19, 18, 33, 40],
            [35, 21, 16, 14, 12, 14, 16, 21, 35],
        ], dtype=PIECE_TYPE_DTYPE)

        # Rank 2 layout (second rank)
        rank2_layout = np.array([
            [23, 30, 36, 31, 3, 31, 36, 30, 23],
            [29, 24, 24, 28, 10, 28, 24, 24, 29],
            [37, 24, 24, 7, 2, 7, 24, 24, 37],
            [5, 25, 38, 4, 17, 4, 38, 25, 5],
            [3, 10, 2, 17, 20, 17, 2, 10, 3],
            [5, 25, 38, 4, 17, 4, 38, 25, 5],
            [37, 24, 24, 7, 2, 7, 24, 24, 37],
            [29, 24, 24, 28, 10, 28, 24, 24, 29],
            [23, 30, 36, 31, 3, 31, 36, 30, 23],
        ], dtype=PIECE_TYPE_DTYPE)

        # Pre-allocate arrays for maximum possible pieces
        max_pieces = 500
        coords = np.empty((max_pieces, 3), dtype=COORD_DTYPE)
        types = np.empty(max_pieces, dtype=PIECE_TYPE_DTYPE)
        colors = np.empty(max_pieces, dtype=COLOR_DTYPE)
        idx = 0

        # Process Rank 1 (z=0 for white, z=8 for black)
        idx = self._place_rank(coords, types, colors, rank1_layout, 0, idx)

        # Process Rank 2 (z=1 for white, z=7 for black)
        idx = self._place_rank(coords, types, colors, rank2_layout, 1, idx)

        # Process Rank 3 - ALL PAWNS (z=2 for white, z=6 for black)
        pawn_type = int(PieceType.PAWN)
        pawn_coords = np.mgrid[0:9, 0:9, 2:3].reshape(3, -1).T.astype(COORD_DTYPE)

        n_pawns = pawn_coords.shape[0]
        coords[idx:idx+n_pawns] = pawn_coords
        types[idx:idx+n_pawns] = pawn_type
        colors[idx:idx+n_pawns] = Color.WHITE
        idx += n_pawns

        # Black pawns at z=6
        pawn_coords_black = pawn_coords.copy()
        pawn_coords_black[:, 2] = 6
        coords[idx:idx+n_pawns] = pawn_coords_black
        types[idx:idx+n_pawns] = pawn_type
        colors[idx:idx+n_pawns] = Color.BLACK
        idx += n_pawns

        # Trim arrays to actual size
        return coords[:idx], types[:idx], colors[:idx]

    def _place_rank(self, coords: np.ndarray, types: np.ndarray, colors: np.ndarray,
                               layout: np.ndarray, z: int, start_idx: int) -> int:
        """Helper to place a single rank."""
        n_squares = layout.size
        y_coords, x_coords = np.divmod(np.arange(n_squares, dtype=INDEX_DTYPE), 9)

        # Filter for non-empty pieces
        mask = layout.flat != 0
        valid_types = layout.flat[mask]
        valid_x = x_coords[mask]
        valid_y = y_coords[mask]
        count = valid_types.size

        # Place white pieces
        end_idx = start_idx + count
        coords[start_idx:end_idx, 0] = valid_x
        coords[start_idx:end_idx, 1] = valid_y
        coords[start_idx:end_idx, 2] = z
        types[start_idx:end_idx] = valid_types
        colors[start_idx:end_idx] = Color.WHITE

        # Place black pieces
        start_idx = end_idx
        end_idx = start_idx + count
        coords[start_idx:end_idx, 0] = valid_x
        coords[start_idx:end_idx, 1] = valid_y
        coords[start_idx:end_idx, 2] = 8 - z
        types[start_idx:end_idx] = valid_types
        colors[start_idx:end_idx] = Color.BLACK
        
        return end_idx

    def copy(self) -> 'Board':
        """Create a copy of the board configuration."""
        new_board = Board()
        new_board.generation = self.generation
        return new_board

    def byte_hash(self) -> int:
        """
        Get a hash of the board state.
        DELEGATES to cache manager if available, otherwise returns 0.
        """
        if self._cache_manager:
            return self._cache_manager.zobrist_cache.current_hash
        return 0

