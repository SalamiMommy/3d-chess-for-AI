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
        """Check if occupied - CONSISTENT (x,y,z) parameters."""
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
        """Batch update positions - INCREMENTAL with priest count fix."""
        for coord, piece in updates:
            x, y, z = coord
            if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
                continue

            old_piece = self.get(coord)
            if old_piece and old_piece.ptype == PieceType.PRIEST:
                self._priest_count[old_piece.color.value] -= 1
            if piece and piece.ptype == PieceType.PRIEST:
                self._priest_count[piece.color.value] += 1

        for coord, piece in updates:
            x, y, z = coord
            if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
                continue

            if piece is None:
                self._occ[z, y, x] = 0
                self._ptype[z, y, x] = 0
            else:
                self._occ[z, y, x] = color_to_code(piece.color)
                self._ptype[z, y, x] = piece.ptype.value

            self._piece_cache.pop(coord, None)

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
        return [coord for coord, piece in pieces.items() if piece.ptype == ptype]

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
