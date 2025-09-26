"""
game3d/board/board.py
9×9×9 board – tensor-first, zero-copy, training-ready."""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Tuple, Iterable, List
from game3d.common.common import (
    SIZE_X, SIZE_Y, SIZE_Z, SIZE, VOLUME, N_TOTAL_PLANES,
    N_COLOR_PLANES, N_PLANES_PER_SIDE, X, Y, Z,
    Coord, in_bounds, coord_to_idx, idx_to_coord,
    tensor_cache, hash_board_tensor,
)
from game3d.pieces.enums import Color, PieceType
from game3d.pieces.piece import Piece

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

    @classmethod
    def startpos(cls) -> "Board":
        """Return a board set up for the initial position."""
        b = cls.empty()   # create an empty board
        b.init_startpos() # fill it (existing instance method)
        return b


    def init_startpos(self) -> None:
        """Set up the initial position with full 9x9 ranks."""

        # Quick name → PieceType lookup
        name_to_pt = {pt.name.lower(): pt for pt in PieceType}
        def parse(name: str) -> PieceType:
            try:
                return name_to_pt[name.lower()]
            except KeyError as e:
                raise ValueError(f"Unknown piece name in template: {name}") from e

        # Helper to place a piece
        def put(x: int, y: int, z: int, pt: PieceType, color: Color) -> None:
            self.set_piece((x, y, z), Piece(color, pt))

        # ------------------------------------------------------------------
        # 1st Rank (z=0 for White, z=8 for Black)
        # ------------------------------------------------------------------
        rank1 = [
            ["reflector", "coneslider", "edgerook", "echo", "orbiter", "echo", "edgerook", "coneslider", "reflector"],
            ["spiral", "xzzigzag", "xzqueen", "yzqueen", "mirror", "yzqueen", "xzqueen", "xzzigzag", "spiral"],
            ["yzzigzag", "friendlyteleporter", "panel", "hive", "knight31", "hive", "panel", "friendlyteleporter", "yzzigzag"],
            ["bomb", "swapper", "nebula", "knight32", "trailblazer", "knight32", "nebula", "swapper", "bomb"],
            ["orbiter", "mirror", "knight31", "trailblazer", "king", "trailblazer", "knight31", "mirror", "orbiter"],
            ["bomb", "swapper", "nebula", "knight32", "trailblazer", "knight32", "nebula", "swapper", "bomb"],
            ["yzzigzag", "friendlyteleporter", "panel", "hive", "knight31", "hive", "panel", "friendlyteleporter", "yzzigzag"],
            ["spiral", "xzzigzag", "xzqueen", "yzqueen", "mirror", "yzqueen", "xzqueen", "xzzigzag", "spiral"],
            ["reflector", "coneslider", "edgerook", "echo", "orbiter", "echo", "edgerook", "coneslider", "reflector"],
        ]

        for y in range(9):
            for x in range(9):
                pt = parse(rank1[y][x])
                put(x, y, 0, pt, Color.WHITE)
                put(x, y, 8, pt, Color.BLACK)

        # ------------------------------------------------------------------
        # 2nd Rank (z=1 for White, z=7 for Black)
        # ------------------------------------------------------------------
        rank2 = [
            ["freezer", "slower", "blackhole", "geomancer", "bishop", "geomancer", "blackhole", "slower", "freezer"],
            ["speeder", "wall", "wall", "armour", "trigonalbishop", "armour", "wall", "wall", "speeder"],
            ["whitehole", "wall", "wall", "priest", "knight", "priest", "wall", "wall", "whitehole"],
            ["queen", "archer", "infiltrator", "rook", "xyqueen", "rook", "infiltrator", "archer", "queen"],
            ["bishop", "trigonalbishop", "knight", "xyqueen", "vectorslider", "xyqueen", "knight", "trigonalbishop", "bishop"],
            ["queen", "archer", "infiltrator", "rook", "xyqueen", "rook", "infiltrator", "archer", "queen"],
            ["whitehole", "wall", "wall", "priest", "knight", "priest", "wall", "wall", "whitehole"],
            ["speeder", "wall", "wall", "armour", "trigonalbishop", "armour", "wall", "wall", "speeder"],
            ["freezer", "slower", "blackhole", "geomancer", "bishop", "geomancer", "blackhole", "slower", "freezer"],
        ]

        for y in range(9):
            for x in range(9):
                pt = parse(rank2[y][x])
                put(x, y, 1, pt, Color.WHITE)
                put(x, y, 7, pt, Color.BLACK)

        # ------------------------------------------------------------------
        # 3rd Rank - Pawns (z=2 for White, z=6 for Black)
        # ------------------------------------------------------------------
        for x in range(9):
            for y in range(9):
                put(x, y, 2, PieceType.PAWN, Color.WHITE)
                put(x, y, 6, PieceType.PAWN, Color.BLACK)

        # ------------------------------------------------------------------
        # z=3,4,5 remain empty
        # ------------------------------------------------------------------
        assert self.validate_tensor(), "Board tensor is invalid after init!"

    def tensor(self) -> torch.Tensor:
        return self._tensor.contiguous()

    def byte_hash(self) -> int:
        if self._hash is None:
            self._hash = hash_board_tensor(self._tensor)
        return self._hash

    def set_piece(self, c: Coord, p: Optional[Piece]) -> None:
        x, y, z = c
        # Optional: assert no piece is already present (for debugging)
        # existing = self.piece_at(c)
        # if existing is not None and p is not None:
        #     print(f"WARNING: Overwriting piece {existing} at {c} with {p}")

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

        # Find indices where value is exactly 1.0 (since we only set 0.0 or 1.0)
        white_one_mask = (white_plane == 1.0)
        if white_one_mask.any():
            idx = int(white_one_mask.nonzero(as_tuple=True)[0][0].item())
            return Piece(Color.WHITE, PieceType(idx))

        black_one_mask = (black_plane == 1.0)
        if black_one_mask.any():
            idx = int(black_one_mask.nonzero(as_tuple=True)[0][0].item())
            return Piece(Color.BLACK, PieceType(idx))

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
    def apply_move(self, mv: Move) -> bool:
        """
        Apply the move to the tensor.
        Returns True  -> move was really executed
               False -> move was ignored (empty square)
        """
        from_coord = mv.from_coord
        to_coord   = mv.to_coord

        piece = self.piece_at(from_coord)
        if piece is None:               # 🔥 guard
            return False                # ← tell caller "nothing happened"

        # ----------  original logic unchanged  ----------
        if mv.is_capture:
            self.set_piece(to_coord, None)

        if getattr(mv, "is_promotion", False) and getattr(mv, "promotion_type", None):
            promoted_piece = Piece(piece.color, mv.promotion_type)
            self.set_piece(from_coord, None)
            self.set_piece(to_coord, promoted_piece)
        else:
            self.set_piece(from_coord, None)
            self.set_piece(to_coord, piece)

        self._hash = None
        return True                     # ← success

    def validate_tensor(self) -> bool:
        """Check that every (x,y,z) has at most one 1.0 in white/black planes."""
        valid = True
        for z in range(SIZE_Z):
            for y in range(SIZE_Y):
                for x in range(SIZE_X):
                    white_vals = self._tensor[WHITE_SLICE, z, y, x]
                    black_vals = self._tensor[BLACK_SLICE, z, y, x]
                    total_ones = (white_vals == 1.0).sum().item() + (black_vals == 1.0).sum().item()
                    if total_ones > 1:
                        print(f"INVALID: Multiple pieces at {(x,y,z)}")
                        print("White:", white_vals)
                        print("Black:", black_vals)
                        valid = False
                    elif total_ones == 0:
                        # empty is fine
                        pass
                    else:
                        # exactly one piece
                        pass
        return valid
