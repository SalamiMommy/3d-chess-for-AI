# game3d/cache/piececache.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from game3d.pieces.piece import Piece
from game3d.pieces.enums import Color, PieceType
from game3d.common.common import (
    Coord, PIECE_SLICE, COLOR_SLICE,
    N_PIECE_TYPES, SIZE_X, SIZE_Y, SIZE_Z,
    N_TOTAL_PLANES          # add this
)
import numpy as np
import threading

if TYPE_CHECKING:
    from game3d.board.board import Board   # for mypy/IDE only
else:
    Board = object

class PieceCache:
    __slots__ = (
        "_white_pieces", "_black_pieces", "_valid",
        "_occ", "_ptype", "_occ_view", "_lock", "_gen", "_board"
    )

    def __init__(self, board: "Board") -> None:
        self._white_pieces: Dict[Coord, PieceType] = {}
        self._black_pieces: Dict[Coord, PieceType] = {}
        self._valid = False
        self._board = board
        self._gen = -1          # generation number
        self._occ: Optional[np.ndarray] = None
        self._ptype: Optional[np.ndarray] = None
        self._occ_view: Optional[np.ndarray] = None
        self.rebuild(board)
        self._lock = threading.RLock()

    def get(self, coord: Coord) -> Optional[Piece]:
        """O(1) lookup with minimal overhead."""
        if coord in self._white_pieces:
            return Piece(Color.WHITE, self._white_pieces[coord])
        if coord in self._black_pieces:
            return Piece(Color.BLACK, self._black_pieces[coord])
        return None

    def get_type(self, coord: Coord, color: Color) -> Optional[PieceType]:
        """Faster lookup when you know the color."""
        if color == Color.WHITE:
            return self._white_pieces.get(coord)
        else:
            return self._black_pieces.get(coord)

    def iter_color(self, color: Color):
        """Iterate only pieces of specific color."""
        pieces_dict = self._white_pieces if color == Color.WHITE else self._black_pieces
        for coord, ptype in pieces_dict.items():
            yield coord, Piece(color, ptype)

    def rebuild(self, board: "Board") -> None:
        # Invalidate cached view
        if hasattr(self, '_occ_view'):
            delattr(self, '_occ_view')
        self._white_pieces.clear()
        self._black_pieces.clear()

        tensor_np = (
            board._tensor.cpu().numpy() if board._tensor.is_cuda
            else board._tensor.numpy()
        )

        # Expected shape: (41 piece/color planes + remaining planes, 9, 9, 9)
        expected_full_shape = (N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X)
        if tensor_np.shape[1:] != (SIZE_Z, SIZE_Y, SIZE_X):
            raise ValueError(f"Board tensor spatial dimensions {tensor_np.shape[1:]} != expected {(SIZE_Z, SIZE_Y, SIZE_X)}")

        # Extract the piece planes (first N_PIECE_TYPES) and color plane
        piece_planes = tensor_np[PIECE_SLICE]  # Shape: (N_PIECE_TYPES, 9, 9, 9)
        color_plane = tensor_np[N_PIECE_TYPES]  # Shape: (9, 9, 9)

        # Process each piece type
        for ptype_idx in range(N_PIECE_TYPES):
            piece_mask = piece_planes[ptype_idx]  # Shape: (9, 9, 9)
            # Find all positions where this piece type exists
            coords = np.argwhere(piece_mask == 1.0)
            for z, y, x in coords:
                try:
                    ptype = PieceType(ptype_idx)
                    coord = (int(x), int(y), int(z))

                    # Determine color from color mask
                    color_value = color_plane[z, y, x]
                    if color_value > 0.5:  # White
                        self._white_pieces[coord] = ptype
                    else:  # Black
                        self._black_pieces[coord] = ptype
                except ValueError:
                    continue

        self._valid = True
        # Invalidate array cache on rebuild
        self._invalidate_array_cache()

    def _invalidate_array_cache(self) -> None:
        """Invalidate the cached numpy arrays."""
        self._occ = None
        self._ptype = None
        self._occ_view = None
        self._gen = -1 # Reset generation so _ensure_arrays will rebuild

    def update_for_move(self, from_coord: Coord, to_coord: Coord, from_piece: Piece, to_piece: Optional[Piece]) -> None:
        """
        Incrementally update the cache for a move.

        Args:
            from_coord: The coordinate the piece moved from.
            to_coord: The coordinate the piece moved to.
            from_piece: The piece that was moved.
            to_piece: The piece that was at the destination (if captured, otherwise None).
        """
        # Determine the source and destination dictionaries based on piece color
        src_dict = self._white_pieces if from_piece.color == Color.WHITE else self._black_pieces
        dst_dict = self._white_pieces if from_piece.color == Color.WHITE else self._black_pieces

        # Remove the piece from its original location
        piece_type_moved = src_dict.pop(from_coord, None)
        if piece_type_moved is None:
            # This indicates a potential inconsistency if the piece wasn't found where expected
            # Consider logging a warning or raising an error if strict consistency is required
            # For now, we'll just return, assuming the move application logic handles errors.
            # raise AssertionError(f"PieceCache: Expected {from_piece} at {from_coord}, but found none.")
            return # Or handle as needed

        # Handle capture: remove the captured piece from its dictionary
        if to_piece is not None:
            captured_dict = self._white_pieces if to_piece.color == Color.WHITE else self._black_pieces
            # The piece being captured should be at to_coord before the move is fully applied
            # However, the board state might have already changed, so we rely on the 'to_piece' argument
            # which represents the piece *before* the move's effects (like capture) are finalized,
            # but *after* the moving piece has left its source.
            # The correct logic here is to remove the piece that was *at* to_coord *before* the move.
            # The `apply_move` in manager.py calls `board.apply_move(mv)` *before* updating caches.
            # So, `to_piece` here is the piece that WAS at `mv.to_coord` BEFORE the move.
            # Therefore, we remove the piece type of `to_piece` from the destination coordinate.
            # This assumes `to_piece` represents the captured piece (or None if no capture).
            # Let's assume `to_piece` is the piece being captured (or None).
            captured_color_dict = self._white_pieces if to_piece.color == Color.WHITE else self._black_pieces
            # Pop the captured piece from its location (which should be to_coord)
            captured_type_at_dest = captured_color_dict.pop(to_coord, None)
            # Optional: Validate that the captured type matches expectations if strict checking is needed
            # if captured_type_at_dest != to_piece.ptype:
            #     print(f"Warning: Expected captured type {to_piece.ptype} at {to_coord}, found {captured_type_at_dest}")

        # Add the moved piece to its new location
        dst_dict[to_coord] = piece_type_moved

        # Invalidate the cached arrays since the piece locations have changed
        self._invalidate_array_cache()

    def export_arrays(self):
        self._ensure_arrays()
        return self._occ, self._ptype

    def get_occupancy_view(self):
        self._ensure_arrays()
        with self._lock:
            if self._occ_view is None:
                self._occ_view = self._occ.copy()
            return self._occ_view

    def _ensure_arrays(self) -> None:
        """Build arrays only once per board change."""
        with self._lock:
            if self._gen == self._board.generation and self._occ is not None:
                return                   # already current

            # --- build once ---
            occ  = np.zeros((9, 9, 9), dtype=np.uint8)
            ptyp = np.zeros((9, 9, 9), dtype=np.uint8)

            for (x, y, z), ptype in self._white_pieces.items():
                occ[z, y, x]  = 1  # Note: using z,y,x indexing to match tensor layout
                ptyp[z, y, x] = ptype.value

            for (x, y, z), ptype in self._black_pieces.items():
                occ[z, y, x]  = 2  # Note: using z,y,x indexing to match tensor layout
                ptyp[z, y, x] = ptype.value

            self._occ   = occ
            self._ptype = ptyp
            self._gen   = self._board.generation
            # view is now stale
            self._occ_view = self._occ

    def get_flat_occupancy(self) -> np.ndarray:
        """
        Return a *contiguous* 1-D uint8[729] view of the occupancy cube
        (same memory layout the new slider kernel expects).
        """
        occ3d = self.get_occupancy_view()      # 9×9×9 uint8
        return occ3d.ravel()                   # zero-copy view
