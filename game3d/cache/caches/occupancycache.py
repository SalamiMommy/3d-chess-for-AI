# occupancycache.py - OPTIMIZED VERSION (Using Common Modules)
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
from game3d.common.piece_utils import color_to_code

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
        self._piece_cache = {}
        self._piece_cache_max_size = 8192
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
            if len(self._piece_cache) < self._piece_cache_max_size:
                self._piece_cache[coord] = None
            return None

        cached = self._piece_cache.get(coord)
        if cached is not None:
            return cached

        color_code = self._occ[z, y, x]
        if color_code == 0:
            if len(self._piece_cache) < self._piece_cache_max_size:
                self._piece_cache[coord] = None
            return None

        color = Color.WHITE if color_code == 1 else Color.BLACK
        ptype = PieceType(self._ptype[z, y, x])
        piece = Piece(color, ptype)

        if len(self._piece_cache) < self._piece_cache_max_size:
            self._piece_cache[coord] = piece

        return piece

    def get_batch(self, coords: np.ndarray) -> list[Piece | None]:
        """Get pieces for multiple coordinates – LOCK-FREE."""
        if coords.size == 0:
            return []

        if coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")

        valid_mask = in_bounds_vectorised(coords)
        if not np.any(valid_mask):
            return [None] * len(coords)

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

        full_results = [None] * len(coords)
        full_results[valid_mask] = results
        return full_results

    def get_type(self, coord: Coord, color: Color) -> Optional[PieceType]:
        """Get piece type - LOCK-FREE."""
        x, y, z = coord
        piece = self.get(coord)
        if piece and piece.color == color:
            return piece.ptype
        return None

    def iter_color(self, color: Optional[Color]) -> Iterator[Tuple[Coord, Piece]]:
        if color is None:
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
        for coord, ptype in occ.items():
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
            self.set_position(coord, piece)
            if piece.color == Color.WHITE:
                self._white_pieces[coord] = piece.ptype
            else:
                self._black_pieces[coord] = piece.ptype
            if piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color] += 1

    def set_position(self, coord: Coord, piece: Optional[Piece]) -> None:
        x, y, z = coord

        if piece is None:
            self._occ[z, y, x] = 0
            self._ptype[z, y, x] = 0
            self._piece_cache.pop(coord, None)
            old_piece = self._piece_cache.get(coord)
            if old_piece and old_piece.ptype == PieceType.PRIEST:
                self._priest_count[old_piece.color] -= 1
            return

        if piece.color not in (Color.WHITE, Color.BLACK):
            raise ValueError(
                f"Illegal colour {piece.color!r} for piece {piece} at {coord}"
            )

        old_piece = self._piece_cache.get(coord)
        if old_piece and old_piece.ptype == PieceType.PRIEST:
            self._priest_count[old_piece.color] -= 1

        colour_code = color_to_code(piece.color)
        self._occ[z, y, x] = colour_code
        self._ptype[z, y, x] = piece.ptype.value

        if len(self._piece_cache) < self._piece_cache_max_size:
            self._piece_cache[coord] = piece

        if piece.ptype == PieceType.PRIEST:
            self._priest_count[piece.color] += 1

    def batch_set_positions(self, updates: List[Tuple[Coord, Optional[Piece]]]) -> None:
        """Vectorized batch update - CRITICAL for 462 pieces."""
        if not updates:
            return

        # Convert to arrays for vectorized processing
        coords = np.array([coord for coord, _ in updates])
        pieces = [piece for _, piece in updates]

        # Batch bounds check
        valid_mask = in_bounds_vectorised(coords)
        if not np.any(valid_mask):
            return

        valid_coords = coords[valid_mask]
        valid_pieces = [pieces[i] for i in range(len(updates)) if valid_mask[i]]

        # Batch update occupancy and piece type arrays
        x_coords, y_coords, z_coords = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]

        # Clear positions first
        self._occ[z_coords, y_coords, x_coords] = 0
        self._ptype[z_coords, y_coords, x_coords] = 0

        # Set new pieces
        for i, (coord, piece) in enumerate(zip(valid_coords, valid_pieces)):
            x, y, z = coord
            if piece is not None:
                self._occ[z, y, x] = color_to_code(piece.color)
                self._ptype[z, y, x] = piece.ptype.value

                # Update piece dictionaries
                if piece.color == Color.WHITE:
                    self._white_pieces[tuple(coord)] = piece.ptype
                else:
                    self._black_pieces[tuple(coord)] = piece.ptype
            else:
                # Remove from dictionaries
                self._white_pieces.pop(tuple(coord), None)
                self._black_pieces.pop(tuple(coord), None)

        # Batch clear piece cache
        for coord in valid_coords:
            self._piece_cache.pop(tuple(coord), None)

        self._valid = True
        self._gen += 1

    def incremental_update(self, moves: List[Tuple[Coord, Coord, Optional[Piece]]]) -> None:
        """Apply incremental updates for multiple moves without full rebuild."""
        updates = []

        for from_coord, to_coord, promotion_piece in moves:
            piece = self.get(from_coord)
            if piece is None:
                continue

            if promotion_piece is not None:
                piece = promotion_piece

            updates.append((from_coord, None))
            updates.append((to_coord, piece))

        self.batch_set_positions(updates)

    def batch_get_pieces(self, coords: List[Coord]) -> List[Optional[Piece]]:
        """Get pieces for multiple coordinates efficiently."""
        coords_array = np.array(coords, dtype=int)
        x_coords, y_coords, z_coords = coords_array.T

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

    def batch_get_types(self, coords: List[Coord], color: Color) -> List[Optional[PieceType]]:
        """Get piece types for multiple coordinates of a specific color."""
        coords_array = np.array(coords, dtype=int)
        x_coords, y_coords, z_coords = coords_array.T

        color_codes = self._occ[z_coords, y_coords, x_coords]
        ptypes = self._ptype[z_coords, y_coords, x_coords]

        expected_code = color_to_code(color)

        results = []
        for color_code, ptype_val in zip(color_codes, ptypes):
            if color_code != expected_code:
                results.append(None)
            else:
                results.append(PieceType(ptype_val))

        return results

    def batch_is_occupied(self, coords: List[Coord]) -> List[bool]:
        """Check if multiple coordinates are occupied."""
        coords_array = np.array(coords, dtype=int)
        z_coords, y_coords, x_coords = coords_array.T

        return (self._occ[z_coords, y_coords, x_coords] != 0).tolist()

    def get_positions_by_type(self, color: Color, ptype: PieceType) -> List[Coord]:
        pieces = self._white_pieces if color == Color.WHITE else self._black_pieces
        return [coord for coord, pt in pieces.items() if pt == ptype]

    def has_piece_type(self, ptype: PieceType, color: Color) -> bool:
        pieces = self._white_pieces if color == Color.WHITE else self._black_pieces
        return any(piece.ptype == ptype for piece in pieces.values())

    def batch_get_occupancy(self, coords: np.ndarray) -> np.ndarray:
        """Vectorized occupancy check for multiple coordinates."""
        if coords.size == 0:
            return np.array([], dtype=bool)

        if coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N,3), got {coords.shape}")

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        x, y, z = clip_coords(x, y, z)
        return self._occ[z, y, x] != 0

    def batch_get_pieces_vectorized(self, coords: np.ndarray) -> List[Optional[Piece]]:
        """Vectorized piece retrieval with caching."""
        if coords.size == 0:
            return []

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

        miss_coords = np.array(cache_misses)
        x, y, z = miss_coords[:, 0], miss_coords[:, 1], miss_coords[:, 2]
        x, y, z = clip_coords(x, y, z)

        color_codes = self._occ[z, y, x]
        ptypes = self._ptype[z, y, x]

        for idx, (coord, color_code, ptype_val) in enumerate(zip(cache_misses, color_codes, ptypes)):
            coord_tuple = tuple(coord)
            if color_code == 0:
                piece = None
            else:
                color = Color.WHITE if color_code == 1 else Color.BLACK
                ptype = PieceType(ptype_val)
                piece = Piece(color, ptype)

            if len(self._piece_cache) < self._piece_cache_max_size:
                self._piece_cache[coord_tuple] = piece

            original_idx = cache_indices[idx]
            results[original_idx] = piece

        return results

    def batch_is_occupied_vectorized(self, coords: np.ndarray) -> np.ndarray:
        """Vectorized occupancy check returning numpy array."""
        if coords.size == 0:
            return np.array([], dtype=bool)

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        x, y, z = clip_coords(x, y, z)
        return self._occ[z, y, x] != 0

    def get_all_piece_coords(self, color: Optional[Color] = None) -> np.ndarray:
        """Get all piece coordinates as numpy array for batch processing."""
        if color is None:
            # All pieces
            white_coords = np.array(list(self._white_pieces.keys()))
            black_coords = np.array(list(self._black_pieces.keys()))
            return np.vstack([white_coords, black_coords]) if len(white_coords) > 0 and len(black_coords) > 0 else (
                white_coords if len(white_coords) > 0 else black_coords
            )
        else:
            pieces = self._white_pieces if color == Color.WHITE else self._black_pieces
            return np.array(list(pieces.keys()))

    def get_pieces_by_types(self, color: Color, types: List[PieceType]) -> np.ndarray:
        """Get coordinates of pieces of specific types - essential for move generation."""
        pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
        type_set = set(types)

        coords = []
        for coord, ptype in pieces_dict.items():
            if ptype in type_set:
                coords.append(coord)

        return np.array(coords) if coords else np.empty((0, 3), dtype=int)

    def iter_color_batch(self, color: Color, batch_size: int = 50) -> Iterator[List[Tuple[Coord, Piece]]]:
        """Batch iteration for move generation - critical for 200+ pieces."""
        pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
        items = list(pieces_dict.items())

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            yield [(coord, Piece(color, ptype)) for coord, ptype in batch]

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
        """Ultra-fast occupancy check assuming pre-validated coordinates."""
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return self._occ[z, y, x] != 0

    def batch_get_colors_and_types(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-fast batch retrieval of colors and types without bounds checking"""
        if coords.size == 0:
            return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        colors = self._occ[z, y, x]
        types = self._ptype[z, y, x]
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
