"""
game3d/board/board.py
9×9×9 board – tensor-first, zero-copy, training-ready."""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Tuple, Iterable
from common import (
    SIZE_X, SIZE_Y, SIZE_Z, SIZE, VOLUME, N_TOTAL_PLANES,
    N_COLOR_PLANES, N_PLANES_PER_SIDE, X, Y, Z,
    Coord, in_bounds, coord_to_idx, idx_to_coord,
    tensor_cache, hash_board_tensor,
)
from pieces.enums import Color, PieceType
from pieces.piece import Piece

# ------------------------------------------------------------------
# Internal layout of the 4-D tensor (C, D, H, W)
# ------------------------------------------------------------------
# channel 0..39      : white pieces (one-hot PieceType)
# channel 40..79     : black pieces (one-hot PieceType)
# channel 80         : current player plane (1 = white, 0 = black)
# ------------------------------------------------------------------
WHITE_SLICE   = slice(0, N_PLANES_PER_SIDE)
BLACK_SLICE   = slice(N_PLANES_PER_SIDE, N_COLOR_PLANES)
CURRENT_SLICE = slice(N_COLOR_PLANES, N_COLOR_PLANES + 1)

class Board:
    """Thin wrapper around a single tensor of shape (N_TOTAL_PLANES, 9, 9, 9)."""

    __slots__ = ("_tensor", "_hash")

    # --------------------------------------------------------------
    # construction
    # --------------------------------------------------------------
    def __init__(self, tensor: Optional[torch.Tensor] = None) -> None:
        if tensor is None:
            self._tensor = torch.zeros(N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X, dtype=torch.float32)
        else:
            if tensor.shape != (N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X):
                raise ValueError("Board tensor must be (C,9,9,9)")
            self._tensor = tensor
        self._hash: Optional[int] = None

    # --------------------------------------------------------------
    # fast factories
    # --------------------------------------------------------------
    @staticmethod
    def empty() -> Board:
        return Board()

    @staticmethod
    def from_fen(fen: str) -> Board:
        """Placeholder – parses custom 9×9×9 FEN later."""
        # TODO: implement when you have a FEN spec
        return Board.empty()

    # --------------------------------------------------------------
    # raw tensor access (zero-copy)
    # --------------------------------------------------------------
    def tensor(self) -> torch.Tensor:
        """Return contiguous tensor (C,9,9,9) – ready for Conv3d."""
        return self._tensor.contiguous()

    def byte_hash(self) -> int:
        """Fast hash for LRU caches."""
        if self._hash is None:
            self._hash = hash_board_tensor(self._tensor)
        return self._hash

    # --------------------------------------------------------------
    # piece-level API (still O(1) because we index a view)
    # --------------------------------------------------------------
    def set_piece(self, c: Coord, p: Optional[Piece]) -> None:
        x, y, z = c
        # zero both colour slices first
        self._tensor[WHITE_SLICE, z, y, x] = 0.0
        self._tensor[BLACK_SLICE, z, y, x] = 0.0
        if p is not None:
            idx = p.ptype
            if p.color == Color.WHITE:
                self._tensor[idx, z, y, x] = 1.0
            else:
                self._tensor[N_PLANES_PER_SIDE + idx, z, y, x] = 1.0
        self._hash = None

    def piece_at(self, c: Coord) -> Optional[Piece]:
        x, y, z = c
        white_plane = self._tensor[WHITE_SLICE, z, y, x]
        black_plane = self._tensor[BLACK_SLICE, z, y, x]
        if white_plane.max() == 1.0:
            return Piece(Color.WHITE, PieceType(int(white_plane.argmax())))
        if black_plane.max() == 1.0:
            return Piece(Color.BLACK, PieceType(int(black_plane.argmax())))
        return None

    def multi_piece_at(self, sq: Tuple[int, int, int]) -> List[Piece]:
        """Share-Square aware – returns **all** pieces on square."""
        from game3d.cache.manager import get_share_square_cache
        return get_share_square_cache().pieces_at(sq)
    # --------------------------------------------------------------
    # bulk operations – fully vectorised
    # --------------------------------------------------------------
    def mirror_z(self) -> Board:
        """Return a new board flipped along z (for data aug)."""
        return Board(self._tensor.flip(dims=(1,)))

    def rotate_90(self, k: int = 1) -> Board:
        """Rotate x-y plane 90° k times (z unchanged)."""
        # rotate dims 2,3
        t = torch.rot90(self._tensor, k, dims=(2, 3))
        return Board(t)

    def apply_player_plane(self, color: Color) -> None:
        """Fill auxiliary plane in-place (1 = white, 0 = black)."""
        self._tensor[CURRENT_SLICE] = float(color)

    # --------------------------------------------------------------
    # numpy / python iteration helpers (only for move gen, not training)
    # --------------------------------------------------------------
    def occupancy_mask(self) -> torch.Tensor:
        """Boolean mask (9,9,9) True if square occupied."""
        return (self._tensor[WHITE_SLICE].sum(axis=0) +
                self._tensor[BLACK_SLICE].sum(axis=0)).bool()

    def list_occupied(self) -> Iterable[Tuple[Coord, Piece]]:
        occ = self.occupancy_mask()
        indices = torch.argwhere(occ)          # (N, 3) tensor
        for z, y, x in indices.tolist():
            c = (x, y, z)
            yield c, self.piece_at(c)

    # --------------------------------------------------------------
    # dataloader integration – zero-copy batch builder
    # --------------------------------------------------------------
    def clone(self) -> Board:
        return Board(self._tensor.clone())

    def share_memory_(self) -> Board:
        """Call before sending to worker processes."""
        self._tensor.share_memory_()
        return self

    # --------------------------------------------------------------
    # repr / debug
    # --------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"Board(tensor={self._tensor.shape}, hash={self.byte_hash()})"
