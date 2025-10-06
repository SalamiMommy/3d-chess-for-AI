from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING  # Removed np.ndarray
import threading
import torch

from game3d.pieces.piece import Piece
from game3d.pieces.enums import Color, PieceType
from game3d.common.common import (
    Coord, PIECE_SLICE, COLOR_SLICE,
    N_PIECE_TYPES, SIZE_X, SIZE_Y, SIZE_Z,
    N_TOTAL_PLANES
)

if TYPE_CHECKING:
    from game3d.board.board import Board

class OccupancyCache:
    __slots__ = (
        "_occ", "_ptype", "_white_pieces", "_black_pieces",
        "_valid", "_occ_view", "_lock", "_gen", "_board",
        "_piece_cache", "_piece_cache_max_size"  # Add these two attributes
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
        self._lock = threading.RLock()
        # Initialize the piece cache
        self._piece_cache = {}
        self._piece_cache_max_size = 1000
        self.rebuild(board)

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """Check if a coordinate is occupied with thread safety."""
        with self._lock:
            return self._occ[z, y, x] != 0

    def is_occupied_batch(self, coords: np.ndarray) -> np.ndarray:
        """Check if multiple coordinates are occupied with thread safety."""
        with self._lock:
            z, y, x = coords.T
            return self._occ[z, y, x] != 0

    @property
    def count(self) -> int:
        return np.count_nonzero(self._occ)

    def tobytes(self):
        return self._occ.tobytes()

    def get(self, coord: Coord) -> Optional[Piece]:
        """Get a piece at the given coordinate with thread safety."""
        with self._lock:
            x, y, z = coord
            color_code = self._occ[z, y, x]
            if color_code == 0:
                return None

            # Check piece cache
            if hasattr(self, '_piece_cache') and coord in self._piece_cache:
                return self._piece_cache[coord]

            color = Color.WHITE if color_code == 1 else Color.BLACK
            ptype = PieceType(self._ptype[z, y, x])
            piece = Piece(color, ptype)

            # Initialize cache if not exists
            if not hasattr(self, '_piece_cache'):
                self._piece_cache = {}
                self._piece_cache_max_size = 1000

            # Update cache if not full
            if len(self._piece_cache) < self._piece_cache_max_size:
                self._piece_cache[coord] = piece

            return piece

    def get_batch(self, coords: np.ndarray) -> List[Optional[Piece]]:
        """Get pieces for multiple coordinates at once with thread safety."""
        with self._lock:
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
        """Get piece type at coordinate for specific color with thread safety."""
        with self._lock:
            x, y, z = coord
            color_code = self._occ[z, y, x]
            expected_code = 1 if color == Color.WHITE else 2
            if color_code != expected_code:
                return None
            return PieceType(self._ptype[z, y, x])

    def iter_color(self, color: Color):
        """Iterate over pieces of a specific color with thread safety."""
        with self._lock:
            pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
            items = list(pieces_dict.items())

        # Yield outside of lock to avoid holding it during iteration
        for coord, ptype in items:
            yield coord, Piece(color, ptype)

    def get_occupancy_view(self):
        with self._lock:
            if self._occ_view is None:
                self._occ_view = self._occ.copy()
            return self._occ_view

    def get_flat_occupancy(self) -> np.ndarray:
        occ3d = self.get_occupancy_view()
        return occ3d.ravel()

    def export_arrays(self):
        """Export occupancy arrays with thread safety."""
        with self._lock:
            return self._occ.copy(), self._ptype.copy()

    def rebuild(self, board: "Board") -> None:
        """Rebuild the occupancy cache with thread safety."""
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

            # Vectorized approach - process all pieces at once
            occupied_mask = piece_planes.sum(axis=0) > 0
            if not np.any(occupied_mask):
                self._valid = True
                self._gen = self._board.generation
                self._occ_view = None
                return

            # Get all coordinates at once
            z_coords, y_coords, x_coords = np.where(occupied_mask)

            # Get piece types for all coordinates
            ptype_indices = np.argmax(piece_planes[:, z_coords, y_coords, x_coords], axis=0)

            # Get color values and determine color codes
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
        """Set a piece at the given coordinate with thread safety."""
        with self._lock:
            x, y, z = coord
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

            # Update piece cache
            self._piece_cache.pop(coord, None)

            self._occ_view = None
            self._gen = self._board.generation

    @property
    def mask(self) -> np.ndarray:
        """Return a 3D boolean mask where True indicates occupied space."""
        with self._lock:
            return self._occ > 0

    # Add these methods to the OccupancyCache class

    def batch_set_positions(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        """
        Set multiple positions in a batch operation for improved performance.

        Args:
            updates: List of tuples containing (coordinate, piece) pairs
        """
        with self._lock:
            # Prepare data structures for batch updates
            batch_final = {}
            for coord, piece in updates:
                batch_final[coord] = piece

            # Update dictionaries
            for coord, new_piece in batch_final.items():
                # Get current piece before update
                current_piece = self.get(coord)

                # Remove existing piece from dictionary if present
                if current_piece is not None:
                    pieces_dict = self._white_pieces if current_piece.color == Color.WHITE else self._black_pieces
                    pieces_dict.pop(coord, None)

                # Add new piece to dictionary if present
                if new_piece is not None:
                    pieces_dict = self._white_pieces if new_piece.color == Color.WHITE else self._black_pieces
                    pieces_dict[coord] = new_piece.ptype

            # Prepare for vectorized array updates
            z_list, y_list, x_list = [], [], []
            occ_vals, ptype_vals = [], []

            for coord, piece in batch_final.items():
                x, y, z = coord
                z_list.append(z)
                y_list.append(y)
                x_list.append(x)

                if piece is None:
                    occ_vals.append(0)
                    ptype_vals.append(0)
                else:
                    occ_vals.append(1 if piece.color == Color.WHITE else 2)
                    ptype_vals.append(piece.ptype.value)

            # Apply vectorized updates if there are changes
            if z_list:
                z_arr = np.array(z_list)
                y_arr = np.array(y_list)
                x_arr = np.array(x_list)
                occ_arr = np.array(occ_vals, dtype=np.uint8)
                ptype_arr = np.array(ptype_vals, dtype=np.uint8)

                self._occ[z_arr, y_arr, x_arr] = occ_arr
                self._ptype[z_arr, y_arr, x_arr] = ptype_arr

            # Update piece cache
            for coord in batch_final:
                self._piece_cache.pop(coord, None)

            # Invalidate view and update generation
            self._occ_view = None
            self._gen = self._board.generation

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
