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
    """Pure numpy board storage - no logic."""
    __slots__ = ('_array', '_cache_manager', 'generation')

    def __init__(self, array=None):
        if array is None:
            self._array = np.zeros((N_TOTAL_PLANES, SIZE, SIZE, SIZE), dtype=FLOAT_DTYPE)
        else:
            self._array = array.astype(FLOAT_DTYPE, copy=False)
        self._cache_manager = None
        self.generation = 0  # Track board modifications for cache invalidation

    @property
    def cache_manager(self):
        """Get the cache manager associated with this board."""
        return self._cache_manager

    @cache_manager.setter
    def cache_manager(self, value):
        """Set the cache manager for this board."""
        self._cache_manager = value

    # CRITICAL: This method is required by many modules
    def array(self) -> np.ndarray:
        """Get underlying board array."""
        return self._array

    def get_board_array(self) -> np.ndarray:
        """Get board array (alias for array())."""
        return self._array

    @staticmethod
    def empty() -> 'Board':
        """Create empty board."""
        return Board()

    @classmethod
    def startpos(cls) -> 'Board':
        """Create board with starting position."""
        b = cls.empty()
        b.init_startpos()
        return b

    def init_startpos(self) -> None:
        """Initialize 3-rank starting position with vectorized operations."""
        self._array[:] = 0.0

        # Define layouts as direct integer arrays (PieceType enum values)
        # This eliminates string parsing entirely

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
            [29, 24, 0, 28, 10, 28, 24, 0, 29],
            [37, 0, 0, 7, 2, 7, 0, 0, 37],
            [5, 25, 38, 4, 17, 4, 38, 25, 5],
            [3, 10, 2, 17, 20, 17, 2, 10, 3],
            [5, 25, 38, 4, 17, 4, 38, 25, 5],
            [37, 24, 0, 7, 2, 7, 0, 24, 37],
            [29, 0, 0, 28, 10, 28, 0, 0, 29],
            [23, 30, 36, 31, 3, 31, 36, 30, 23],
        ], dtype=PIECE_TYPE_DTYPE)

        # Pre-allocate arrays for maximum possible pieces
        max_pieces = 500
        coords = np.empty((max_pieces, 3), dtype=COORD_DTYPE)
        types = np.empty(max_pieces, dtype=PIECE_TYPE_DTYPE)
        colors = np.empty(max_pieces, dtype=COLOR_DTYPE)
        idx = 0

        # Process Rank 1 (z=0 for white, z=8 for black)
        self._vectorized_place_rank(coords, types, colors, rank1_layout, 0, idx)
        idx += rank1_layout.size * 2  # *2 for both colors

        # Process Rank 2 (z=1 for white, z=7 for black)
        self._vectorized_place_rank(coords, types, colors, rank2_layout, 1, idx)
        idx += rank2_layout.size * 2

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
        coords = coords[:idx]
        types = types[:idx]
        colors = colors[:idx]

        # Batch place all pieces
        if len(coords) > 0:
            self._place_pieces_vectorized(coords, types, colors)
            self.generation += 1

    def _vectorized_place_rank(self, coords: np.ndarray, types: np.ndarray, colors: np.ndarray,
                               layout: np.ndarray, z: int, start_idx: int) -> None:
        """Vectorized placement of a single rank."""
        n_squares = layout.size
        y_coords, x_coords = np.divmod(np.arange(n_squares, dtype=INDEX_DTYPE), 9)

        # Place white pieces
        end_idx = start_idx + n_squares
        coords[start_idx:end_idx, 0] = x_coords
        coords[start_idx:end_idx, 1] = y_coords
        coords[start_idx:end_idx, 2] = z
        types[start_idx:end_idx] = layout.flat
        colors[start_idx:end_idx] = Color.WHITE

        # Place black pieces
        start_idx = end_idx
        end_idx = start_idx + n_squares
        coords[start_idx:end_idx, 0] = x_coords
        coords[start_idx:end_idx, 1] = y_coords
        coords[start_idx:end_idx, 2] = 8 - z
        types[start_idx:end_idx] = layout.flat
        colors[start_idx:end_idx] = Color.BLACK

    def _place_pieces_vectorized(self, coords: np.ndarray, piece_types: np.ndarray, colors: np.ndarray) -> None:
        """Place pieces at coordinates - fully vectorized."""
        if coords.shape[0] == 0:
            return

        # Convert color values to plane offsets (0 for white, 1 for black)
        color_offsets = (colors - Color.WHITE).astype(INDEX_DTYPE)
        plane_indices = (piece_types - 1) + (color_offsets * N_PIECE_TYPES)

        # Ensure correct dtypes for indexing
        plane_indices = plane_indices.astype(INDEX_DTYPE)
        x = coords[:, 0].astype(INDEX_DTYPE)
        y = coords[:, 1].astype(INDEX_DTYPE)
        z = coords[:, 2].astype(INDEX_DTYPE)

        # Vectorized assignment
        self._array[plane_indices, x, y, z] = 1.0

    def get_piece_at(self, coord) -> tuple:
        """Vectorized piece lookup. Returns (piece_type, color) or (None, None)"""
        coord_arr = np.asarray(coord, dtype=COORD_DTYPE).reshape(3)
        x, y, z = coord_arr

        if in_bounds_vectorized(coord_arr.reshape(1, 3))[0]:
            planes = self._array[:, x, y, z]
            plane_idx = np.argmax(planes)
            if planes[plane_idx] > 0:
                color = COLOR_WHITE if plane_idx < N_PIECE_TYPES else COLOR_BLACK
                piece_type = (plane_idx % N_PIECE_TYPES) + 1
                return piece_type, color
        return None, None

    def get_pieces_at_vectorized(self, coords: np.ndarray) -> np.ndarray:
        """Public interface with LOUD error detection."""
        coords_arr = validate_coords(coords)

        if coords_arr.shape[0] > VECTORIZATION_THRESHOLD:
            results = self._batch_get_pieces_optimized(self._array, coords_arr)
        else:
            results = self._batch_get_pieces_fallback(coords_arr)

        # LOUD FAILURE: Check for sentinel values
        if np.any(results == -1.0):
            invalid_indices = np.where(results == -1.0)[0]
            logger.critical(f"🚨 NUMBA PROCESSED INVALID COORDINATES: {coords_arr[invalid_indices]}")
            raise ValueError(f"Numba bounds error at indices: {invalid_indices}")

        return results

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def _batch_get_pieces_optimized(board: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Optimized batch retrieval with bounds checking."""
        assert board.ndim == 4, f"Board must be 4D, got {board.ndim}"
        assert board.shape[0] == N_TOTAL_PLANES, f"Board planes mismatch: {board.shape[0]} != {N_TOTAL_PLANES}"
        assert coords.ndim == 2, f"Coords must be 2D, got {coords.ndim}"
        assert coords.shape[1] == 3, f"Coords must be (N,3), got {coords.shape}"

        n_coords = coords.shape[0]
        results = np.empty(n_coords, dtype=FLOAT_DTYPE)

        for i in prange(n_coords):
            x, y, z = coords[i]
            if 0 <= x <= MAX_COORD_VALUE and 0 <= y <= MAX_COORD_VALUE and 0 <= z <= MAX_COORD_VALUE:
                results[i] = board[:, x, y, z].max()
            else:
                results[i] = -1.0  # Sentinel for invalid

        return results

    def _batch_get_pieces_fallback(self, coords: np.ndarray) -> np.ndarray:
        """Fallback for small batches."""
        results = np.empty(coords.shape[0], dtype=FLOAT_DTYPE)
        for i, (x, y, z) in enumerate(coords):
            if 0 <= x <= MAX_COORD_VALUE and 0 <= y <= MAX_COORD_VALUE and 0 <= z <= MAX_COORD_VALUE:
                results[i] = self._array[:, x, y, z].max()
            else:
                results[i] = -1.0
        return results

    def set_piece_at(self, coord, piece_type: int, color: int) -> None:
        """Vectorized piece placement with generation tracking."""
        coord_arr = np.asarray(coord, dtype=COORD_DTYPE).reshape(3)
        x, y, z = coord_arr

        if in_bounds_vectorized(coord_arr.reshape(1, 3))[0]:
            self._array[:, x, y, z] = 0.0
            if piece_type > 0:
                # Calculate plane index
                color_offset = 0 if color == Color.WHITE else N_PIECE_TYPES
                plane_idx = (piece_type - 1) + color_offset
                if 0 <= plane_idx < N_TOTAL_PLANES:
                    self._array[plane_idx, x, y, z] = 1.0
            self.generation += 1  # CRITICAL: Track board modifications

    def batch_set_pieces_at(self, coords: np.ndarray, piece_types: np.ndarray, colors: np.ndarray) -> None:
        """Set multiple pieces at once - increments generation only once."""
        if coords.shape[0] == 0:
            return

        # Validate all coordinates
        valid_mask = in_bounds_vectorized(coords)
        valid_coords = coords[valid_mask]
        valid_types = piece_types[valid_mask]
        valid_colors = colors[valid_mask]

        if len(valid_coords) == 0:
            return

        # Clear all target squares first
        for i in range(len(valid_coords)):
            x, y, z = valid_coords[i]
            self._array[:, x, y, z] = 0.0

        # Set new pieces
        color_offsets = (valid_colors - Color.WHITE).astype(INDEX_DTYPE)
        plane_indices = (valid_types - 1) + (color_offsets * N_PIECE_TYPES)

        plane_indices = plane_indices.astype(INDEX_DTYPE)
        x = valid_coords[:, 0].astype(INDEX_DTYPE)
        y = valid_coords[:, 1].astype(INDEX_DTYPE)
        z = valid_coords[:, 2].astype(INDEX_DTYPE)

        self._array[plane_indices, x, y, z] = 1.0
        self.generation += 1

    def copy(self) -> 'Board':
        """Create a copy of the board."""
        new_board = Board(self._array.copy())
        new_board.generation = self.generation
        return new_board

    def byte_hash(self) -> int:
        """Get a hash of the board state for repetition detection."""
        # Use a fast hash that can handle numpy arrays
        return hash(self._array.tobytes())
