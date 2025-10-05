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
        "_valid", "_occ_view", "_lock", "_gen", "_board"
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
        self.rebuild(board)

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        return self._occ[z, y, x] != 0

    def is_occupied_batch(self, coords: np.ndarray) -> np.ndarray:
        z, y, x = coords.T
        return self._occ[z, y, x] != 0

    @property
    def count(self) -> int:
        return np.count_nonzero(self._occ)

    def tobytes(self):
        return self._occ.tobytes()

    def get(self, coord: Coord) -> Optional[Piece]:
        x, y, z = coord
        color_code = self._occ[z, y, x]
        if color_code == 0:
            return None
        color = Color.WHITE if color_code == 1 else Color.BLACK
        ptype = PieceType(self._ptype[z, y, x])
        return Piece(color, ptype)

    def get_batch(self, coords: np.ndarray) -> List[Optional[Piece]]:
        """Get pieces for multiple coordinates at once."""
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
        x, y, z = coord
        color_code = self._occ[z, y, x]
        expected_code = 1 if color == Color.WHITE else 2
        if color_code != expected_code:
            return None
        return PieceType(self._ptype[z, y, x])

    def iter_color(self, color: Color):
        pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
        for coord, ptype in pieces_dict.items():
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
        return self._occ, self._ptype

    def rebuild(self, board: "Board") -> None:
        self._white_pieces.clear()
        self._black_pieces.clear()
        self._occ.fill(0)
        self._ptype.fill(0)

        tensor_np = board._tensor.numpy() if isinstance(board._tensor, torch.Tensor) else board._tensor
        piece_planes = tensor_np[PIECE_SLICE]
        color_plane = tensor_np[N_PIECE_TYPES]

        # Find all occupied positions at once
        for ptype_idx in range(N_PIECE_TYPES):
            piece_mask = piece_planes[ptype_idx] == 1.0
            if not np.any(piece_mask):
                continue

            # Get all coordinates for this piece type
            coords = np.argwhere(piece_mask)

            # Get color values for all these coordinates
            color_values = color_plane[coords[:, 0], coords[:, 1], coords[:, 2]]

            # Determine color codes (1 for white, 2 for black)
            color_codes = np.where(color_values > 0.5, 1, 2)

            # Update occupancy and piece type arrays
            z_coords, y_coords, x_coords = coords.T
            self._occ[z_coords, y_coords, x_coords] = color_codes
            self._ptype[z_coords, y_coords, x_coords] = ptype_idx

            # Update piece dictionaries
            white_mask = color_codes == 1
            black_mask = color_codes == 2

            white_coords = coords[white_mask]
            black_coords = coords[black_mask]

            for coord in white_coords:
                self._white_pieces[tuple(coord[::-1])] = PieceType(ptype_idx)  # Reverse to (x,y,z)

            for coord in black_coords:
                self._black_pieces[tuple(coord[::-1])] = PieceType(ptype_idx)  # Reverse to (x,y,z)

        self._valid = True
        self._gen = self._board.generation
        self._occ_view = None

    def set_position(self, coord: Coord, piece: Optional[Piece]) -> None:
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
        self._occ_view = None
        self._gen = self._board.generation

    @property
    def mask(self) -> np.ndarray:
        """Return a 3D boolean mask where True indicates occupied space."""
        return self._occ > 0
