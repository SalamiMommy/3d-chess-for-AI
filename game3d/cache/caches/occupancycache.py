# occupancycache.py - FIXED VERSION (Consistent coordinate handling)
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, List, Iterator
import torch

from game3d.pieces.piece import Piece
from game3d.common.enums import Color, PieceType
from game3d.common.constants import (
    Coord, PIECE_SLICE, COLOR_SLICE,
    N_PIECE_TYPES, SIZE_X, SIZE_Y, SIZE_Z,
    N_TOTAL_PLANES
)
from game3d.common.coord_utils import clip_coords, filter_valid_coords, in_bounds, in_bounds_vectorised

class OccupancyCache:
    """Optimized occupancy cache with minimal locking overhead."""
    __slots__ = (
        "_occ", "_ptype", "_white_pieces", "_black_pieces",
        "_valid", "_occ_view", "_lock", "_gen", "_board",
        "_piece_cache", "_piece_cache_max_size", "_priest_count"
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
        # Aggressive caching
        self._piece_cache = {}
        self._piece_cache_max_size = 8192  # Larger cache
        self._priest_count = np.zeros(2, dtype=np.uint8)
        self.rebuild(board)
        # CORRECTION: Add assert for color codes
        unique = np.unique(self._occ)
        if not np.all(np.isin(unique, [0, 1, 2])):
            bad = unique[~np.isin(unique, [0, 1, 2])]
            raise AssertionError(
                f"Occupancy array contains illegal colour code(s) {bad.tolist()}. "
                f"Only [0,1,2] are allowed (0=empty, 1=white, 2=black)."
            )

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """Check if occupied - NO LOCK for read-only operations."""
        return self._occ[z, y, x] != 0

    def is_occupied_batch(self, coords: np.ndarray) -> np.ndarray:
        """Batch check – NO LOCK, with defensive clamp."""
        if coords.size == 0:
            return np.array([], dtype=bool)

        # FIXED: Consistent coordinate handling - expect (x,y,z) input
        if coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        x, y, z = clip_coords(x, y, z)  # UPDATED: Use common.py
        return self._occ[z, y, x] != 0

    def has_priest(self, color: Color) -> bool:
        """Lock-free priest check."""
        return self._priest_count[color] > 0

    def any_priest_alive(self) -> bool:
        """True if at least one priest of any color is on the board."""
        return self._priest_count.any()

    @property
    def count(self) -> int:
        return np.count_nonzero(self._occ)

    def tobytes(self):
        return self._occ.tobytes()

    def get(self, coord: Coord) -> Optional[Piece]:
        """Get piece - LOCK-FREE cache hit, lock only on miss."""
        # FIXED: Consistent coordinate handling
        x, y, z = coord  # Expect (x,y,z)

        # Early exit for OOB coordinates (defensive)
        if not in_bounds(coord):
            # Optionally cache None for this invalid coord (rare, but prevents repeated checks)
            if len(self._piece_cache) < self._piece_cache_max_size:
                self._piece_cache[coord] = None
            return None

        # Try cache first WITHOUT lock
        cached = self._piece_cache.get(coord)
        if cached is not None:
            return cached

        # Cache miss - safe to index now
        color_code = self._occ[z, y, x]
        if color_code == 0:
            # Cache the None result
            if len(self._piece_cache) < self._piece_cache_max_size:
                self._piece_cache[coord] = None
            return None

        # Construct and cache piece...
        color = Color.WHITE if color_code == 1 else Color.BLACK
        ptype = PieceType(self._ptype[z, y, x])
        piece = Piece(color, ptype)

        # Cache it WITHOUT lock
        if len(self._piece_cache) < self._piece_cache_max_size:
            self._piece_cache[coord] = piece

        return piece

    def get_batch(self, coords: np.ndarray) -> list[Piece | None]:
        """
        Get pieces for multiple coordinates – LOCK-FREE.
        Gracefully handles OOB inputs by clamping.
        """
        if coords.size == 0:
            return []

        # FIXED: Consistent coordinate handling
        if coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")

        # Filter OOB upfront (fast vectorised reject)
        valid_mask = in_bounds_vectorised(coords)
        if not np.any(valid_mask):
            return [None] * len(coords)

        # Clamp the valid coordinates before indexing
        valid_coords = coords[valid_mask]
        x_coords, y_coords, z_coords = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]
        x_coords, y_coords, z_coords = clip_coords(x_coords, y_coords, z_coords)

        color_codes = self._occ[z_coords, y_coords, x_coords]
        ptypes = self._ptype[z_coords, y_coords, x_coords]

        results = []
        for i in range(len(color_codes)):
            if color_codes[i] == 0:
                results.append(None)
            else:
                color = Color.WHITE if color_codes[i] == 1 else Color.BLACK
                ptype = PieceType(ptypes[i])
                results.append(Piece(color, ptype))

        # Pad results for invalid coords
        full_results = [None] * len(coords)
        full_results[valid_mask] = results
        return full_results

    def get_type(self, coord: Coord, color: Color) -> Optional[PieceType]:
        """Get piece type - LOCK-FREE."""
        # FIXED: Consistent coordinate handling
        x, y, z = coord
        piece = self.get(coord)
        if piece and piece.color == color:
            return piece.ptype
        return None

    def _clip_coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clamp coordinates to board bounds."""
        return np.clip(x, 0, SIZE_X - 1), np.clip(y, 0, SIZE_Y - 1), np.clip(z, 0, SIZE_Z - 1)

    def iter_color(self, color: Optional[Color]) -> Iterator[Tuple[Coord, Piece]]:
        if color is None:                       # white-hole special case
            for c, pt in self._white_pieces.items():
                yield c, Piece(Color.WHITE, pt)
            for c, pt in self._black_pieces.items():
                yield c, Piece(Color.BLACK, pt)
            return

        occ = self._white_pieces if color == Color.WHITE else self._black_pieces
        for coord, ptype in occ.items():
            yield coord, Piece(color, ptype)

    def find_king(self, color: Color) -> Optional[Coord]:
        occ = self._white_pieces if color == Color.WHITE else self._black_pieces
        for coord, ptype in occ.items():          # ← ptype is a PieceType enum
            if ptype == PieceType.KING:
                return coord
        return None

    def rebuild(self, board: "Board") -> None:
        """Full rebuild of the occupancy cache."""
        self._occ.fill(0)
        self._ptype.fill(0)
        self._white_pieces.clear()
        self._black_pieces.clear()
        self._piece_cache.clear()
        self._priest_count.fill(0)
        self._valid = True
        self._gen = getattr(board, 'generation', 0)

        for coord, piece in board.list_occupied():
            self.set_position(coord, piece)   # <<< this already updates _occ, _ptype
            # >>>  MISSING: keep the dicts in sync  <<<
            if piece.color == Color.WHITE:
                self._white_pieces[coord] = piece.ptype
            else:
                self._black_pieces[coord] = piece.ptype
            if piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color] += 1

    def set_position(self, coord: Coord, piece: Optional[Piece]) -> None:
        # FIXED: Consistent coordinate handling
        x, y, z = coord

        if piece is None:
            self._occ[z, y, x] = 0
            self._ptype[z, y, x] = 0
            self._piece_cache.pop(coord, None)
            # update priest count for removed piece
            old_piece = self._piece_cache.get(coord)
            if old_piece and old_piece.ptype == PieceType.PRIEST:
                self._priest_count[old_piece.color] -= 1
            return

        # >>>>>>  NEW: reject garbage colour  <<<<<<
        if piece.color not in (Color.WHITE, Color.BLACK):
            raise ValueError(
                f"Illegal colour {piece.color!r} for piece {piece} at {coord}"
            )

        # Update priest count for removed piece
        old_piece = self._piece_cache.get(coord)
        if old_piece and old_piece.ptype == PieceType.PRIEST:
            self._priest_count[old_piece.color] -= 1

        colour_code = 1 if piece.color == Color.WHITE else 2
        self._occ[z, y, x] = colour_code
        self._ptype[z, y, x] = piece.ptype.value
        # Cache the piece
        if len(self._piece_cache) < self._piece_cache_max_size:
            self._piece_cache[coord] = piece
        # Update priest count
        if piece.ptype == PieceType.PRIEST:
            self._priest_count[piece.color] += 1

    def batch_set_positions(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        """
        Batch update positions - INCREMENTAL with priest count fix.
        """
        # CORRECTION: Update priest counts before applying
        for coord, piece in updates:
            old_piece = self.get(coord)  # Cache hit, but lock held now
            if old_piece and old_piece.ptype == PieceType.PRIEST:
                self._priest_count[old_piece.color.value] -= 1
            if piece and piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color.value] += 1

        # Apply updates (standard logic)
        for coord, piece in updates:
            x, y, z = coord
            if piece is None:
                self._occ[z, y, x] = 0
                self._ptype[z, y, x] = 0
            else:
                self._occ[z, y, x] = 1 if piece.color == Color.WHITE else 2
                self._ptype[z, y, x] = piece.ptype.value
            # CORRECTION: Invalidate piece cache for affected coords
            self._piece_cache.pop(coord, None)

        self._valid = True
        self._gen += 1

    def incremental_update(self, moves: List[Tuple[Coord, Coord, Optional[Piece]]]) -> None:
        """
        Apply incremental updates for multiple moves without full rebuild.

        Args:
            moves: List of tuples containing (from_coord, to_coord, promotion_piece)
                promotion_piece is Optional[Piece] for pawn promotions
        """

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

        # Convert list to numpy array for vectorized operations
        coords_array = np.array(coords, dtype=int)
        z_coords, y_coords, x_coords = coords_array.T

        # Vectorized occupancy check
        return (self._occ[z_coords, y_coords, x_coords] != 0).tolist()

    def get_positions_by_type(self, color: Color, ptype: PieceType) -> List[Coord]:
        pieces = self._white_pieces if color == Color.WHITE else self._black_pieces
        return [coord for coord, piece in pieces.items() if piece.ptype == ptype]

    def has_piece_type(self, ptype: PieceType, color: Color) -> bool:
        pieces = self._white_pieces if color == Color.WHITE else self._black_pieces
        return any(piece.ptype == ptype for piece in pieces.values())

    # Add these methods to the OccupancyCache class

    def batch_get_occupancy(self, coords: np.ndarray) -> np.ndarray:
        """Vectorized occupancy check for multiple coordinates."""
        if coords.size == 0:
            return np.array([], dtype=bool)

        # Consistent coordinate handling: expect (x,y,z)
        if coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        x, y, z = self._clip_coords(x, y, z)
        return self._occ[z, y, x] != 0

    def batch_get_pieces_vectorized(self, coords: np.ndarray) -> List[Optional[Piece]]:
        """Vectorized piece retrieval with caching."""
        if coords.size == 0:
            return []

        # Check cache first
        results = []
        cache_misses = []
        cache_indices = []

        for i, coord in enumerate(coords):
            coord_tuple = tuple(coord)
            cached = self._piece_cache.get(coord_tuple)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                cache_misses.append(coord)
                cache_indices.append(i)

        if not cache_misses:
            return results

        # Process cache misses in batch
        miss_coords = np.array(cache_misses)
        x, y, z = miss_coords[:, 0], miss_coords[:, 1], miss_coords[:, 2]
        x, y, z = self._clip_coords(x, y, z)

        color_codes = self._occ[z, y, x]
        ptypes = self._ptype[z, y, x]

        # Fill cache and results
        for idx, (coord, color_code, ptype_val) in enumerate(zip(cache_misses, color_codes, ptypes)):
            coord_tuple = tuple(coord)
            if color_code == 0:
                piece = None
            else:
                color = Color.WHITE if color_code == 1 else Color.BLACK
                ptype = PieceType(ptype_val)
                piece = Piece(color, ptype)

            # Cache the result
            if len(self._piece_cache) < self._piece_cache_max_size:
                self._piece_cache[coord_tuple] = piece

            # Update results
            original_idx = cache_indices[idx]
            results[original_idx] = piece

        return results

    def batch_is_occupied_vectorized(self, coords: np.ndarray) -> np.ndarray:
        """Vectorized occupancy check returning numpy array."""
        if coords.size == 0:
            return np.array([], dtype=bool)

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        x, y, z = self._clip_coords(x, y, z)
        return self._occ[z, y, x] != 0
