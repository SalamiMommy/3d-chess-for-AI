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
        """
        Set up the initial position.
        Ranks run along the Z-axis (z=0…8).
        - z=1 and z=7: pawns
        - z=2 and z=8: second-rank quadrant (given template)
        - z=3 and z=7: first-rank quadrant (previous template) – shifted to z=3/7 so
                    that the 5-deep block is centred in the 9-layer board.
        All quadrants are centred in their X-Y planes:
            X: centre 5 columns → base_x = 2
            Y: centre 5 rows    → base_y = 2
        """

        # ------------------------------------------------------------------
        # 1.  Quick name → PieceType lookup
        # ------------------------------------------------------------------
        name_to_pt = {pt.name.lower(): pt for pt in PieceType}

        def parse(name: str) -> PieceType:
            try:
                return name_to_pt[name.lower()]
            except KeyError as e:
                raise ValueError(f"Unknown piece name in template: {name}") from e

        # ------------------------------------------------------------------
        # 2.  Quadrant geometry (centred 5×5 window)
        # ------------------------------------------------------------------
        BASE_X, BASE_Y = 2, 2          # top-left of the 5×5 block
        QUAD_SIZE = 5

        # ------------------------------------------------------------------
        # 3.  Helper to place a piece
        # ------------------------------------------------------------------
        def put(x: int, y: int, z: int, pt: PieceType, color: Color) -> None:
            self.set_piece((x, y, z), Piece(color, pt))

        # ------------------------------------------------------------------
        # 4.  First-rank quadrant (old template) → z=0 (white) / z=9 (black)
        # ------------------------------------------------------------------
        rank1_quad = [
            ["reflector", "coneslider", "edgerook", "echo", "orbiter"],
            ["spiral", "xzzigzag", "xzqueen", "xyqueen", "trigonalbishop"],
            ["yzzigzag", "friendlyteleporter", "panel", "hive", "knight31"],
            ["bomb", "swapper", "nebula", "knight32", "friendlyteleporter"],
            ["orbiter", "trigonalbishop", "knight31", "friendlyteleporter", "king"],
        ]

        for dy in range(QUAD_SIZE):
            for dx in range(QUAD_SIZE):
                x = BASE_X + dx
                y = BASE_Y + dy
                pt = parse(rank1_quad[dy][dx])
                put(x, y, 0, pt, Color.WHITE)
                put(x, y, 8, pt, Color.BLACK)

        # ------------------------------------------------------------------
        # 5.  Second-rank quadrant (new template) → z=1 (white) / z=8 (black)
        # ------------------------------------------------------------------
        rank2_quad = [
            ["freezer", "slower", "blackhole", "geomancer", "bishop"],
            ["speeder", "wall", "wall", "armour", "trigonalbishop"],
            ["whitehole", "wall", "wall", "priest", "knight"],
            ["queen", "archer", "infiltrator", "rook", "xyqueen"],
            ["bishop", "trigonalbishop", "knight", "xyqueen", "vectorslider"],
        ]

        for dy in range(QUAD_SIZE):
            for dx in range(QUAD_SIZE):
                x = BASE_X + dx
                y = BASE_Y + dy
                pt = parse(rank2_quad[dy][dx])
                put(x, y, 1, pt, Color.WHITE)
                put(x, y, 7, pt, Color.BLACK)

        # ------------------------------------------------------------------
        # 6.  Pawns on third rank (z=2 white / z=7 black)
        # ------------------------------------------------------------------
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                self.set_piece((x, y, 2), Piece(Color.WHITE, PieceType.PAWN))
                self.set_piece((x, y, 6), Piece(Color.BLACK, PieceType.PAWN))

        # ------------------------------------------------------------------
        # 7.  Leave z=3,4,5 empty – ready for play
        # ------------------------------------------------------------------


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

        w_max = white_plane.max().item()
        if w_max > 0.5:
            return Piece(Color.WHITE, PieceType(int(white_plane.argmax())))

        b_max = black_plane.max().item()
        if b_max > 0.5:
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

