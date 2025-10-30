# occupancycache.py
# occupancycache.py - OPTIMIZED VERSION (Using Common Modules)
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, List, Iterator
import torch
from functools import lru_cache

from game3d.pieces.piece import Piece
from game3d.common.enums import Color, PieceType
from game3d.common.constants import (
    Coord, PIECE_SLICE, COLOR_SLICE,
    N_PIECE_TYPES, SIZE_X, SIZE_Y, SIZE_Z,
    N_TOTAL_PLANES
)
from game3d.common.coord_utils import clip_coords, filter_valid_coords, in_bounds, in_bounds_vectorised
from game3d.common.piece_utils import color_to_code

class OccupancyCache:
    __slots__ = (
        "_occ", "_ptype", "_white_pieces", "_black_pieces",
        "_valid", "_occ_view", "_lock", "_gen", "_manager",
        "_piece_cache", "_priest_count", "_board"
    )

    def __init__(self, manager: "OptimizedCacheManager") -> None:
        self._manager = manager
        board = manager.board

        # Use uint8 for everything - most efficient for GPU transfer
        self._occ = np.zeros((9, 9, 9), dtype=np.uint8)  # 0=empty, 1=white, 2=black
        self._ptype = np.zeros((9, 9, 9), dtype=np.uint8)  # PieceType values

        self._white_pieces: Dict[Coord, PieceType] = {}
        self._black_pieces: Dict[Coord, PieceType] = {}
        self._valid = False
        self._board = board
        self._gen = -1
        self._piece_cache = {}
        self._priest_count = np.zeros(2, dtype=np.uint8)
        self.rebuild(board)

        unique = np.unique(self._occ)
        if not np.all(np.isin(unique, [0, 1, 2])):
            bad = unique[~np.isin(unique, [0, 1, 2])]
            raise AssertionError(
                f"Occupancy array contains illegal colour code(s) {bad.tolist()}. "
                f"Only [0,1,2] are allowed (0=empty, 1=white, 2=black)."
            )

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """Check if occupied - parameters are (x,y,z), array indexed as [z,y,x]."""
        return self._occ[z, y, x] != 0

    def is_occupied_batch(self, coords: np.ndarray) -> np.ndarray:
        """Batch check – NO LOCK, with defensive clamp."""
        if coords.size == 0:
            return np.array([], dtype=bool)

        if coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        x, y, z = clip_coords(x, y, z)
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
        """Get piece - CONSISTENT (x,y,z) coordinate input."""
        x, y, z = coord

        if not in_bounds(coord):
            self._piece_cache[coord] = None
            return None

        cached = self._piece_cache.get(coord)
        if cached is not None:
            return cached

        color_code = self._occ[z, y, x]
        if color_code == 0:
            self._piece_cache[coord] = None
            return None

        color = Color.WHITE if color_code == 1 else Color.BLACK
        ptype = PieceType(self._ptype[z, y, x])
        piece = Piece(color, ptype)


        self._piece_cache[coord] = piece

        return piece

    def get_batch(self, coords: np.ndarray, skip_validation: bool = False, return_raw: bool = False) -> list[Piece | None]:
        """Get pieces for multiple coordinates – LOCK-FREE."""
        if coords.size == 0:
            return []

        if coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")

        if not skip_validation:
            valid_mask = in_bounds_vectorised(coords)
            if not np.any(valid_mask):
                return [None] * len(coords)
            valid_coords = coords[valid_mask]
        else:
            valid_mask = None
            valid_coords = coords

        x_coords, y_coords, z_coords = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]
        x_coords, y_coords, z_coords = clip_coords(x_coords, y_coords, z_coords)

        color_codes = self._occ[z_coords, y_coords, x_coords]
        ptypes = self._ptype[z_coords, y_coords, x_coords]

        if return_raw:
            return color_codes, ptypes

        results = []
        for i in range(len(color_codes)):
            if color_codes[i] == 0:
                results.append(None)
            else:
                color = Color.WHITE if color_codes[i] == 1 else Color.BLACK
                ptype = PieceType(ptypes[i])
                results.append(Piece(color, ptype))

        if skip_validation:
            return results
        else:
            # FIX: Convert boolean mask to integer indices for list assignment
            full_results = [None] * len(coords)
            valid_indices = np.where(valid_mask)[0]
            for idx, result in zip(valid_indices, results):
                full_results[idx] = result
            return full_results

    def get_type(self, x: int, y: int, z: int) -> int:
        """Get piece type code."""
        return self._ptype[z, y, x]

    def get_color(self, x: int, y: int, z: int) -> int:
        """Get color code: 0 empty, 1 white, 2 black."""
        return self._occ[z, y, x]

    def get_color_batch(self, coords: np.ndarray) -> np.ndarray:
        """Batch get color codes."""
        if coords.size == 0:
            return np.array([])

        valid_mask = in_bounds_vectorised(coords)
        valid_coords = coords[valid_mask]
        x, y, z = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]
        x, y, z = clip_coords(x, y, z)
        colors = self._occ[z, y, x]

        full_colors = np.zeros(len(coords), dtype=np.uint8)
        full_colors[valid_mask] = colors
        return full_colors

    def get_type_batch(self, coords: np.ndarray) -> np.ndarray:
        """Batch get piece types."""
        if coords.size == 0:
            return np.array([])

        valid_mask = in_bounds_vectorised(coords)
        valid_coords = coords[valid_mask]
        x, y, z = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]
        x, y, z = clip_coords(x, y, z)
        types = self._ptype[z, y, x]

        full_types = np.zeros(len(coords), dtype=np.uint8)
        full_types[valid_mask] = types
        return full_types

    def iter_occupied(self) -> Iterator[Tuple[Coord, Piece]]:
        """Iterate occupied squares with Piece instances."""
        for z in range(SIZE_Z):
            for y in range(SIZE_Y):
                for x in range(SIZE_X):
                    color_code = self._occ[z, y, x]
                    if color_code != 0:
                        color = Color.WHITE if color_code == 1 else Color.BLACK
                        ptype = PieceType(self._ptype[z, y, x])
                        yield (x, y, z), Piece(color, ptype)

    def iter_color(self, color: Color) -> Iterator[Tuple[Coord, Piece]]:
        """Iterate pieces of specific color."""
        pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
        for coord_tuple, ptype in pieces_dict.items():
            yield coord_tuple, Piece(color, ptype)

    def iter_batch(self, coords: np.ndarray) -> Iterator[Tuple[Coord, Optional[Piece]]]:
        """Batch iterator for given coordinates."""
        colors, types = self.batch_get_piece_attributes(coords)
        for i, coord in enumerate(coords):
            cc, pt = colors[i], types[i]
            if cc == 0:
                yield tuple(coord), None
            else:
                piece_color = Color.WHITE if cc == 1 else Color.BLACK
                yield tuple(coord), Piece(piece_color, PieceType(pt))

    def rebuild(self, board: "Board") -> None:
        """Rebuild from board."""
        self._occ.fill(0)
        self._ptype.fill(0)
        self._white_pieces.clear()
        self._black_pieces.clear()
        self._priest_count = np.zeros(2, dtype=np.int8)
        self._piece_cache.clear()

        for coord, piece in board.enumerate_occupied():
            x, y, z = coord
            color_code = 1 if piece.color == Color.WHITE else 2
            self._occ[z, y, x] = color_code
            self._ptype[z, y, x] = piece.ptype.value
            pieces_dict = self._white_pieces if piece.color == Color.WHITE else self._black_pieces
            pieces_dict[coord] = piece.ptype

            if piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color.value] += 1

        self._valid = True
        self._gen = board.generation if hasattr(board, 'generation') else self._gen + 1

    def update(self, coord: Coord, piece: Optional[Piece]) -> None:
        """Update single square."""
        x, y, z = coord
        coord_tuple = tuple(coord)

        # Update priest count
        old_piece = self.get(coord)
        if old_piece and old_piece.ptype == PieceType.PRIEST:
            self._priest_count[old_piece.color.value] -= 1
        if piece and piece.ptype == PieceType.PRIEST:
            self._priest_count[piece.color.value] += 1

        # Update arrays and dicts
        if piece is None:
            self._occ[z, y, x] = 0
            self._ptype[z, y, x] = 0
            self._white_pieces.pop(coord_tuple, None)
            self._black_pieces.pop(coord_tuple, None)
        else:
            color_code = 1 if piece.color == Color.WHITE else 2
            self._occ[z, y, x] = color_code
            self._ptype[z, y, x] = piece.ptype.value
            pieces_dict = self._white_pieces if piece.color == Color.WHITE else self._black_pieces
            pieces_dict[coord_tuple] = piece.ptype

        self._piece_cache.pop(coord_tuple, None)
        self._gen += 1

    def batch_update(self, batch: List[Tuple[Coord, Optional[Piece]]]) -> None:
        """Batch update occupancy."""
        coords = np.array([coord for coord, _ in batch], dtype=np.int32)
        pieces = [piece for _, piece in batch]

        # Update priest counts - use int8 for consistency
        old_pieces = self.get_batch(coords)
        priest_delta = np.zeros(2, dtype=np.int8)  # Changed to int8
        for old, new in zip(old_pieces, pieces):
            if old and old.ptype == PieceType.PRIEST:
                priest_delta[old.color.value] -= 1
            if new and new.ptype == PieceType.PRIEST:
                priest_delta[new.color.value] += 1

        # Now both arrays are int8, so no casting issues
        self._priest_count = np.clip(self._priest_count + priest_delta, 0, 4)

        # Vectorized array updates
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # Clear all coordinates first
        self._occ[z, y, x] = 0
        self._ptype[z, y, x] = 0

        # Update non-None pieces
        coords_arr = np.array([c for c, p in batch if p is not None])
        pieces_arr = [p for c, p in batch if p is not None]
        if len(coords_arr) > 0:
            x, y, z = coords_arr[:, 0], coords_arr[:, 1], coords_arr[:, 2]
            colors = np.array([1 if p.color == Color.WHITE else 2 for p in pieces_arr])
            types = np.array([p.ptype.value for p in pieces_arr])
            self._occ[z, y, x] = colors
            self._ptype[z, y, x] = types

        # Update dictionaries using a simpler approach
        for coord, piece in batch:
            coord_tuple = tuple(coord)

            # Remove from both dictionaries
            self._white_pieces.pop(coord_tuple, None)
            self._black_pieces.pop(coord_tuple, None)
            self._piece_cache.pop(coord_tuple, None)

            # Add to appropriate dictionary if piece exists
            if piece is not None:
                if piece.color == Color.WHITE:
                    self._white_pieces[coord_tuple] = piece.ptype
                else:
                    self._black_pieces[coord_tuple] = piece.ptype

        self._gen += 1

    def batch_invalidate_cache(self, coords: np.ndarray) -> None:
        """Mass cache invalidation for move generation."""
        if coords.size == 0:
            return

        for coord in coords:
            self._piece_cache.pop(tuple(coord), None)

    def batch_update_priest_count(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        """Batch update priest counts without per-piece function calls."""
        priest_delta = np.zeros(2, dtype=np.int8)

        for coord, new_piece in updates:
            old_piece = self._piece_cache.get(tuple(coord))

            # Old piece was priest
            if old_piece and old_piece.ptype == PieceType.PRIEST:
                priest_delta[old_piece.color.value] -= 1

            # New piece is priest
            if new_piece and new_piece.ptype == PieceType.PRIEST:
                priest_delta[new_piece.color.value] += 1

        # Apply delta
        self._priest_count = np.clip(self._priest_count + priest_delta, 0, None)

    # Add to occupancycache.py
    @lru_cache(maxsize=32)  # Cache based on board gen or Zobrist
    def batch_get_all_pieces_data(self, color: Optional[Color] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get all piece coordinates and types as arrays for maximum vectorization."""
        if color is None:
            # Combine both colors - critical for 462 pieces
            white_coords = np.array(list(self._white_pieces.keys()), dtype=np.int32)
            black_coords = np.array(list(self._black_pieces.keys()), dtype=np.int32)

            if len(white_coords) == 0:
                all_coords = black_coords
                all_types = np.array([self._black_pieces[tuple(coord)] for coord in black_coords], dtype=np.uint8)
            elif len(black_coords) == 0:
                all_coords = white_coords
                all_types = np.array([self._white_pieces[tuple(coord)] for coord in white_coords], dtype=np.uint8)
            else:
                all_coords = np.vstack([white_coords, black_coords])
                white_types = np.array([self._white_pieces[tuple(coord)] for coord in white_coords], dtype=np.uint8)
                black_types = np.array([self._black_pieces[tuple(coord)] for coord in black_coords], dtype=np.uint8)
                all_types = np.concatenate([white_types, black_types])
        else:
            pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
            all_coords = np.array(list(pieces_dict.keys()), dtype=np.int32)
            all_types = np.array([pieces_dict[tuple(coord)] for coord in all_coords], dtype=np.uint8)

        return all_coords, all_types

    def batch_get_piece_attributes(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized piece attribute retrieval - no bounds checking for speed."""
        if coords.size == 0:
            return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        colors = self._occ[z, y, x]
        types = self._ptype[z, y, x]
        return colors, types

    def batch_is_occupied_fast(self, coords: np.ndarray) -> np.ndarray:
        """Optimized occupancy check."""
        if coords.size == 0:
            return np.array([], dtype=bool)

        n = len(coords)
        result = np.zeros(n, dtype=bool)
        occ_flat = self._occ.ravel()

        for i in range(n):
            x, y, z = coords[i]
            if 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9:
                idx = z * 81 + y * 9 + x
                result[i] = occ_flat[idx] != 0

        return result

    def batch_get_colors_and_types(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ULTRA-OPTIMIZED batch retrieval using direct memory access."""
        if coords.size == 0:
            return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

        n = len(coords)

        # Pre-allocate arrays
        colors = np.empty(n, dtype=np.uint8)
        types = np.empty(n, dtype=np.uint8)

        # Get direct references to avoid repeated attribute lookups
        occ = self._occ
        ptype = self._ptype

        # Single-pass processing with bounds checking
        for i in range(n):
            x, y, z = coords[i]
            # Manual bounds checking (faster than function calls)
            if 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9:
                colors[i] = occ[z, y, x]
                types[i] = ptype[z, y, x]
            else:
                colors[i] = 0
                types[i] = 0

        return colors, types

    def batch_update_piece_dicts(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        """Batch update piece dictionaries for white and black pieces"""
        for coord, piece in updates:
            coord_tuple = tuple(coord)

            # Remove from both dictionaries first
            self._white_pieces.pop(coord_tuple, None)
            self._black_pieces.pop(coord_tuple, None)

            # Add to appropriate dictionary if piece exists
            if piece:
                if piece.color == Color.WHITE:
                    self._white_pieces[coord_tuple] = piece.ptype
                else:
                    self._black_pieces[coord_tuple] = piece.ptype

    def get_batch_raw(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # colors, types, valid_mask
        if coords.size == 0:
            return np.array([]), np.array([]), np.array([])

        valid_mask = in_bounds_vectorised(coords)
        valid_coords = coords[valid_mask]
        x, y, z = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]
        x, y, z = clip_coords(x, y, z)  # Assuming clip_coords vectorized

        colors = self._occ[z, y, x]
        types = self._ptype[z, y, x]
        return colors, types, valid_mask

    def incremental_update(self, updates: List[Tuple[Coord, Coord, Optional[Piece]]]) -> None:
        """Incremental update for move operations - handles (from_coord, to_coord, piece) tuples"""
        batch_updates = []

        for from_coord, to_coord, piece in updates:
            # Remove piece from from_coord
            batch_updates.append((from_coord, None))

            # Set piece at to_coord (if piece is provided)
            if piece is not None:
                batch_updates.append((to_coord, piece))
            else:
                # For cases where we don't have the piece, get it from the cache
                moving_piece = self.get(from_coord)
                if moving_piece:
                    batch_updates.append((to_coord, moving_piece))

        if batch_updates:
            self.batch_update(batch_updates)
