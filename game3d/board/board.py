"""
game3d/board/board.py
9×9×9 board – tensor-first, zero-copy, training-ready."""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Tuple, Iterable, List
from common import (
    SIZE_X, SIZE_Y, SIZE_Z, SIZE, VOLUME, N_TOTAL_PLANES,
    N_COLOR_PLANES, N_PLANES_PER_SIDE, X, Y, Z,
    Coord, in_bounds, coord_to_idx, idx_to_coord,
    tensor_cache, hash_board_tensor,
)
from pieces.enums import Color, PieceType
from pieces.piece import Piece

WHITE_SLICE   = slice(0, N_PLANES_PER_SIDE)
BLACK_SLICE   = slice(N_PLANES_PER_SIDE, N_COLOR_PLANES)
CURRENT_SLICE = slice(N_COLOR_PLANES, N_COLOR_PLANES + 1)

class Board:
    """Thin wrapper around a single tensor of shape (N_TOTAL_PLANES, 9, 9, 9)."""

    __slots__ = ("_tensor", "_hash")

    def __init__(self, tensor: Optional[torch.Tensor] = None) -> None:
        if tensor is None:
            self._tensor = torch.zeros(N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X, dtype=torch.float32)
        else:
            if tensor.shape != (N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X):
                raise ValueError("Board tensor must be (C,9,9,9)")
            self._tensor = tensor
        self._hash: Optional[int] = None

    @staticmethod
    def empty() -> Board:
        return Board()

    @staticmethod
    def from_fen(fen: str) -> Board:
        # TODO: implement when you have a FEN spec
        return Board.empty()

    def tensor(self) -> torch.Tensor:
        return self._tensor.contiguous()

    def byte_hash(self) -> int:
        if self._hash is None:
            self._hash = hash_board_tensor(self._tensor)
        return self._hash

    def set_piece(self, c: Coord, p: Optional[Piece]) -> None:
        x, y, z = c
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
        from game3d.cache.manager import get_share_square_cache
        return get_share_square_cache().pieces_at(sq)

    def mirror_z(self) -> Board:
        return Board(self._tensor.flip(dims=(1,)))

    def rotate_90(self, k: int = 1) -> Board:
        t = torch.rot90(self._tensor, k, dims=(2, 3))
        return Board(t)

    def apply_player_plane(self, color: Color) -> None:
        self._tensor[CURRENT_SLICE] = float(color)

    def occupancy_mask(self) -> torch.Tensor:
        return (self._tensor[WHITE_SLICE].sum(axis=0) +
                self._tensor[BLACK_SLICE].sum(axis=0)).bool()

    def list_occupied(self) -> Iterable[Tuple[Coord, Piece]]:
        occ = self.occupancy_mask()
        indices = torch.argwhere(occ)
        for z, y, x in indices.tolist():
            c = (x, y, z)
            yield c, self.piece_at(c)

    def clone(self) -> Board:
        return Board(self._tensor.clone())

    def share_memory_(self) -> Board:
        self._tensor.share_memory_()
        return self

    def __repr__(self) -> str:
        return f"Board(tensor={self._tensor.shape}, hash={self.byte_hash()})"

    # --------------------------------------------------------------
    # Move execution (NEW)
    # --------------------------------------------------------------
    def apply_move(self, mv) -> None:
        """
        Apply the move to the tensor.
        mv: Move object with from_coord, to_coord, is_capture, is_promotion, etc.
        """
        from_coord = mv.from_coord
        to_coord = mv.to_coord
        piece = self.piece_at(from_coord)
        if piece is None:
            raise ValueError(f"No piece to move at {from_coord}")

        # Handle capture: remove target piece at to_coord
        if mv.is_capture:
            self.set_piece(to_coord, None)

        # Handle promotion (if applicable)
        if hasattr(mv, "is_promotion") and mv.is_promotion and hasattr(mv, "promotion_type") and mv.promotion_type:
            promoted_piece = Piece(piece.color, mv.promotion_type)
            self.set_piece(from_coord, None)
            self.set_piece(to_coord, promoted_piece)
        else:
            self.set_piece(from_coord, None)
            self.set_piece(to_coord, piece)

        # En passant, castling, etc. can be handled here if implemented

        self._hash = None

    def undo_move(self, mv) -> None:
        """
        Undo the move on the tensor.
        mv: Move object with from_coord, to_coord, is_capture, etc.
        """
        from_coord = mv.from_coord
        to_coord = mv.to_coord

        # Get the moved piece
        piece = self.piece_at(to_coord)
        if piece is None:
            # If move was a promotion, the promoted piece would be here; revert to pawn
            if hasattr(mv, "is_promotion") and mv.is_promotion and hasattr(mv, "promotion_type") and mv.promotion_type:
                original_piece = Piece(piece.color, PieceType.PAWN)
                self.set_piece(from_coord, original_piece)
                self.set_piece(to_coord, None)
                return
            raise ValueError(f"No piece to undo move at {to_coord}")

        # Revert promotion
        if hasattr(mv, "is_promotion") and mv.is_promotion and hasattr(mv, "promotion_type") and mv.promotion_type:
            original_piece = Piece(piece.color, PieceType.PAWN)
            self.set_piece(from_coord, original_piece)
            self.set_piece(to_coord, None)
        else:
            self.set_piece(from_coord, piece)
            # Restore captured piece (if any)
            if hasattr(mv, "captured_piece") and mv.captured_piece:
                captured_color = piece.color.opposite()
                captured_piece = Piece(captured_color, mv.captured_piece)
                self.set_piece(to_coord, captured_piece)
            else:
                self.set_piece(to_coord, None)

        self._hash = None
