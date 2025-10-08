# occupancycache.py - FIXED VERSION
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, List
import threading
import torch

from game3d.pieces.piece import Piece
from game3d.pieces.enums import Color, PieceType
from game3d.common.common import (
    Coord, PIECE_SLICE, COLOR_SLICE,
    N_PIECE_TYPES, SIZE_X, SIZE_Y, SIZE_Z,
    N_TOTAL_PLANES
)

class OccupancyCache:
    """Optimized occupancy cache with minimal locking overhead."""
    __slots__ = (
        "_occ", "_ptype", "_white_pieces", "_black_pieces",
        "_valid", "_occ_view", "_lock", "_gen", "_board",
        "_piece_cache", "_piece_cache_max_size"
    )

    def __init__(self, board: "Board") -> None:
        self._occ = np.zeros((SIZE_Z, SIZE_Y, SIZE_X), dtype=np.uint8)
        self._ptype = np.zeros((SIZE_Z, SIZE_Y, SIZE_X), dtype=np.uint8)
        self._white_pieces: Dict[Coord, PieceType] = {}
        self._black_pieces: Dict[Coord, PieceType] = {}
        self._valid = False
        self._board = board
        self._gen = -1
        self._occ_view: Optional[np.ndarray] = None
        # Use a simpler lock that doesn't reenter
        self._lock = threading.Lock()
        # Aggressive caching
        self._piece_cache = {}
        self._piece_cache_max_size = 8192  # Larger cache
        self.rebuild(board)

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """Check if occupied - NO LOCK for read-only operations."""
        return self._occ[z, y, x] != 0

    def is_occupied_batch(self, coords: np.ndarray) -> np.ndarray:
        """Batch check - NO LOCK."""
        z, y, x = coords.T
        return self._occ[z, y, x] != 0

    @property
    def count(self) -> int:
        return np.count_nonzero(self._occ)

    def tobytes(self):
        return self._occ.tobytes()

    def get(self, coord: Coord) -> Optional[Piece]:
        """Get piece - LOCK-FREE cache hit, lock only on miss."""
        # Try cache first WITHOUT lock
        cached = self._piece_cache.get(coord)
        if cached is not None:
            return cached

        # Cache miss - need to construct piece
        x, y, z = coord
        color_code = self._occ[z, y, x]
        if color_code == 0:
            # Cache the None result
            if len(self._piece_cache) < self._piece_cache_max_size:
                self._piece_cache[coord] = None
            return None

        # Construct piece
        color = Color.WHITE if color_code == 1 else Color.BLACK
        ptype = PieceType(self._ptype[z, y, z])
        piece = Piece(color, ptype)

        # Cache it WITHOUT lock (dict assignment is atomic in CPython)
        if len(self._piece_cache) < self._piece_cache_max_size:
            self._piece_cache[coord] = piece

        return piece

    def get_batch(self, coords: np.ndarray) -> List[Optional[Piece]]:
        """Get pieces for multiple coordinates - LOCK-FREE."""
        x_coords, y_coords, z_coords = coords.T
        color_codes = self._occ[z_coords, y_coords, x_coords]
        ptypes = self._ptype[z_coords, y_coords, x_coords]

        results = []
        for color_code, ptype_val in zip(color_codes, ptypes):
            if color_code == 0:
                results.append(None)
            else:
                color = Color.WHITE if color_code == 1 else Color.BLACK
                ptype = PieceType(ptype_val)
                results.append(Piece(color, ptype))

        return results

    def get_type(self, coord: Coord, color: Color) -> Optional[PieceType]:
        """Get piece type - LOCK-FREE."""
        x, y, z = coord
        color_code = self._occ[z, y, x]
        expected_code = 1 if color == Color.WHITE else 2
        if color_code != expected_code:
            return None
        return PieceType(self._ptype[z, y, x])

    def iter_color(self, color: Color):
        """Iterate over pieces - snapshot WITHOUT holding lock."""
        # Make a snapshot of the dictionary WITHOUT lock
        pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
        # Dict iteration is thread-safe for reading in CPython
        items = list(pieces_dict.items())

        # Yield outside of lock
        for coord, ptype in items:
            yield coord, Piece(color, ptype)

    def get_occupancy_view(self):
        """Get occupancy view - LOCK-FREE for reads."""
        if self._occ_view is None:
            self._occ_view = self._occ.copy()
        return self._occ_view

    def get_flat_occupancy(self) -> np.ndarray:
        """Get flat occupancy - LOCK-FREE."""
        occ3d = self.get_occupancy_view()
        return occ3d.ravel()

    def export_arrays(self):
        """Export arrays - LOCK-FREE copy."""
        return self._occ.copy(), self._ptype.copy()

    def rebuild(self, board: "Board") -> None:
        """Rebuild cache - this is the ONLY method that needs a lock."""
        with self._lock:
            self._white_pieces.clear()
            self._black_pieces.clear()
            self._occ.fill(0)
            self._ptype.fill(0)

            # Clear piece cache
            self._piece_cache.clear()

            tensor_np = board._tensor.numpy() if isinstance(board._tensor, torch.Tensor) else board._tensor
            piece_planes = tensor_np[PIECE_SLICE]
            color_plane = tensor_np[N_PIECE_TYPES]

            # Vectorized approach
            occupied_mask = piece_planes.sum(axis=0) > 0
            if not np.any(occupied_mask):
                self._valid = True
                self._gen = self._board.generation
                self._occ_view = None
                return

            # Get all coordinates at once
            z_coords, y_coords, x_coords = np.where(occupied_mask)

            # Get piece types
            plane_data = piece_planes[:, z_coords, y_coords, x_coords]
            ptype_indices = np.uint8(np.argmax(plane_data, axis=0))

            # Get color values
            color_values = color_plane[z_coords, y_coords, x_coords]
            color_codes = np.where(color_values > 0.5, 1, 2)

            # Update arrays in bulk
            self._occ[z_coords, y_coords, x_coords] = color_codes
            self._ptype[z_coords, y_coords, x_coords] = ptype_indices

            # Update dictionaries using vectorized operations
            white_mask = color_codes == 1
            black_mask = color_codes == 2

            # Process white pieces
            if np.any(white_mask):
                white_z, white_y, white_x = z_coords[white_mask], y_coords[white_mask], x_coords[white_mask]
                white_ptypes = ptype_indices[white_mask]
                self._white_pieces.update({
                    (x, y, z): PieceType(ptype)
                    for x, y, z, ptype in zip(white_x, white_y, white_z, white_ptypes)
                })

            # Process black pieces
            if np.any(black_mask):
                black_z, black_y, black_x = z_coords[black_mask], y_coords[black_mask], x_coords[black_mask]
                black_ptypes = ptype_indices[black_mask]
                self._black_pieces.update({
                    (x, y, z): PieceType(ptype)
                    for x, y, z, ptype in zip(black_x, black_y, black_z, black_ptypes)
                })

            self._valid = True
            self._gen = self._board.generation
            self._occ_view = None

    def set_position(self, coord: Coord, piece: Optional[Piece]) -> None:
        """Set position - minimal locking."""
        x, y, z = coord

        # Update main structures atomically
        if piece is None:
            color_code = self._occ[z, y, x]
            if color_code == 0:
                return
            pieces_dict = self._white_pieces if color_code == 1 else self._black_pieces
            pieces_dict.pop(coord, None)
            self._occ[z, y, x] = 0
            self._ptype[z, y, x] = 0
        else:
            color_code = 1 if piece.color == Color.WHITE else 2
            pieces_dict = self._white_pieces if color_code == 1 else self._black_pieces
            pieces_dict[coord] = piece.ptype
            self._occ[z, y, x] = color_code
            self._ptype[z, y, x] = piece.ptype.value

        # Invalidate cache entry (atomic)
        self._piece_cache.pop(coord, None)
        self._occ_view = None
        self._gen = self._board.generation

    @property
    def mask(self) -> np.ndarray:
        """Return occupancy mask - LOCK-FREE."""
        return self._occ > 0

    def batch_set_positions(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        """Batch update - lock only once."""
        # Prepare updates
        coords_to_clear = []
        coords_to_set = []

        for coord, piece in updates:
            if piece is None:
                coords_to_clear.append(coord)
            else:
                coords_to_set.append((coord, piece))

        # Apply all updates
        for coord in coords_to_clear:
            self.set_position(coord, None)

        for coord, piece in coords_to_set:
            self.set_position(coord, piece)

    def incremental_update(self, moves: List[Tuple[Coord, Coord, Optional[Piece]]]) -> None:
        """
        Apply incremental updates for multiple moves without full rebuild.

        Args:
            moves: List of tuples containing (from_coord, to_coord, promotion_piece)
                promotion_piece is Optional[Piece] for pawn promotions
        """
        with self._lock:
            # Process all moves in batch
            updates = []

            for from_coord, to_coord, promotion_piece in moves:
                # Get piece at from_coord
                piece = self.get(from_coord)
                if piece is None:
                    continue  # Skip if no piece at source

                # Handle promotion if specified
                if promotion_piece is not None:
                    piece = promotion_piece

                # Prepare updates: remove from source, add to destination
                updates.append((from_coord, None))
                updates.append((to_coord, piece))

            # Apply all updates in batch
            self.batch_set_positions(updates)

    def batch_get_pieces(self, coords: List[Coord]) -> List[Optional[Piece]]:
        """
        Get pieces for multiple coordinates efficiently.

        Args:
            coords: List of coordinates to query

        Returns:
            List of pieces (None for empty squares)
        """
        with self._lock:
            # Convert list to numpy array for vectorized operations
            coords_array = np.array(coords, dtype=int)
            x_coords, y_coords, z_coords = coords_array.T

            # Get color codes and piece types in vectorized manner
            color_codes = self._occ[z_coords, y_coords, x_coords]
            ptypes = self._ptype[z_coords, y_coords, x_coords]

            # Process results
            results = []
            for color_code, ptype_val in zip(color_codes, ptypes):
                if color_code == 0:
                    results.append(None)
                else:
                    color = Color.WHITE if color_code == 1 else Color.BLACK
                    ptype = PieceType(ptype_val)
                    results.append(Piece(color, ptype))

            return results

    def batch_get_types(self, coords: List[Coord], color: Color) -> List[Optional[PieceType]]:
        """
        Get piece types for multiple coordinates of a specific color.

        Args:
            coords: List of coordinates to query
            color: Color of pieces to check

        Returns:
            List of piece types (None for non-matching squares)
        """
        with self._lock:
            # Convert list to numpy array for vectorized operations
            coords_array = np.array(coords, dtype=int)
            x_coords, y_coords, z_coords = coords_array.T

            # Get color codes and piece types in vectorized manner
            color_codes = self._occ[z_coords, y_coords, x_coords]
            ptypes = self._ptype[z_coords, y_coords, x_coords]

            # Expected color code
            expected_code = 1 if color == Color.WHITE else 2

            # Process results
            results = []
            for color_code, ptype_val in zip(color_codes, ptypes):
                if color_code != expected_code:
                    results.append(None)
                else:
                    results.append(PieceType(ptype_val))

            return results

    def batch_is_occupied(self, coords: List[Coord]) -> List[bool]:
        """
        Check if multiple coordinates are occupied.

        Args:
            coords: List of coordinates to check

        Returns:
            List of booleans indicating occupancy
        """
        with self._lock:
            # Convert list to numpy array for vectorized operations
            coords_array = np.array(coords, dtype=int)
            z_coords, y_coords, x_coords = coords_array.T

            # Vectorized occupancy check
            return (self._occ[z_coords, y_coords, x_coords] != 0).tolist()
