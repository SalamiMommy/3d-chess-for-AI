# game3d/board/board.py
"""
9×9×9 board – tensor-first, zero-copy, training-ready.
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Tuple, Iterable, List
from game3d.common.coord_utils import  Coord, in_bounds, coord_to_idx, idx_to_coord
from game3d.common.tensor_utils import hash_board_tensor
from game3d.common.piece_utils import iterate_occupied
from game3d.common.constants import SIZE_X, SIZE_Y, SIZE_Z, SIZE, VOLUME, N_PIECE_TYPES, PIECE_SLICE, COLOR_SLICE, CURRENT_SLICE, EFFECT_SLICE, N_CHANNELS, N_COLOR_PLANES, N_TOTAL_PLANES
from game3d.common.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.board.symmetry import SymmetryManager
from game3d.movement.movepiece import Move

class Board:
    __slots__ = (
        "_tensor", "_hash", "_symmetry_manager",
        "_occupancy_mask", "_occupied_list", "_gen", "cache_manager", "__weakref__"
    )

    def __init__(self, tensor: Optional[torch.Tensor] = None) -> None:
        if tensor is None:
            self._tensor = torch.zeros(N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X, dtype=torch.float32)
        else:
            if tensor.shape != (N_TOTAL_PLANES, SIZE_Z, SIZE_Y, SIZE_X):
                raise ValueError("Board tensor must be (C,9,9,9)")
            self._tensor = tensor
        self._hash: Optional[int] = None
        self._occupancy_mask: Optional[torch.Tensor] = None
        self._occupied_list: Optional[List[Tuple[Coord, Piece]]] = None
        self._gen = 0
        self._symmetry_manager: Optional[SymmetryManager] = None
        self.cache_manager = None

    def _ensure_symmetry_manager(self) -> SymmetryManager:
        """Lazy initialization of symmetry manager to avoid circular imports."""
        if self._symmetry_manager is None:
            self._symmetry_manager = SymmetryManager()
        return self._symmetry_manager

    @property
    def symmetry_manager(self) -> SymmetryManager:
        """Get symmetry manager, initializing lazily if needed."""
        return self._ensure_symmetry_manager()

    def get_symmetric_variants(self) -> List[Tuple[str, 'Board']]:
        """Get all symmetric variants of current board position."""
        return self.symmetry_manager.get_symmetric_boards(self)

    def get_canonical_form(self) -> Tuple['Board', str]:
        """Get canonical (normalized) form of board position."""
        return self.symmetry_manager.get_canonical_form(self)

    def is_symmetric_to(self, other: 'Board') -> bool:
        """Check if this board is symmetrically equivalent to another."""
        return self.symmetry_manager.is_symmetric_position(self, other)

    @staticmethod
    def empty() -> Board:
        return Board()

    def init_startpos(self) -> None:
        """Set up the initial position with full 9x9 ranks."""

        # Quick name → PieceType lookup
        name_to_pt = {pt.name.lower(): pt for pt in PieceType}
        def parse(name: str | None) -> PieceType | None:
            if name is None:               # empty square
                return None
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
                if pt is not None:
                    put(x, y, 0, pt, Color.WHITE)
                    put(x, y, 8, pt, Color.BLACK)

        # ------------------------------------------------------------------
        # 2nd Rank (z=1 for White, z=7 for Black)
        # ------------------------------------------------------------------
        rank2 = [
            ["freezer", "slower", "blackhole", "geomancer", "bishop", "geomancer", "blackhole", "slower", "freezer"],
            ["speeder", "wall", None, "armour", "trigonalbishop", "armour", "wall", None, "speeder"],
            ["whitehole", None, None, "priest", "knight", "priest", None, None, "whitehole"],
            ["queen", "archer", "infiltrator", "rook", "xyqueen", "rook", "infiltrator", "archer", "queen"],
            ["bishop", "trigonalbishop", "knight", "xyqueen", "vectorslider", "xyqueen", "knight", "trigonalbishop", "bishop"],
            ["queen", "archer", "infiltrator", "rook", "xyqueen", "rook", "infiltrator", "archer", "queen"],
            ["whitehole", "wall", None, "priest", "knight", "priest", "wall", None, "whitehole"],
            ["speeder", None, None, "armour", "trigonalbishop", "armour", None, None, "speeder"],
            ["freezer", "slower", "blackhole", "geomancer", "bishop", "geomancer", "blackhole", "slower", "freezer"],
        ]

        for y in range(9):
            for x in range(9):
                pt = parse(rank2[y][x])
                if pt is not None:          # only place a piece when the square is not empty
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
        self._gen += 1
        # NEW: if a cache manager is already attached, refresh it
        if self.cache_manager is not None:
            self.cache_manager.occupancy.rebuild(self)

        # print("[POST-INIT] BK plane-40 value:", self._tensor[40, 8, 4, 4].item())
        # print("[POST-INIT] BK nonzero:", self._tensor[:, 8, 4, 4].nonzero(as_tuple=False).flatten())


    @property
    def occupancy_mask(self) -> torch.Tensor:
        if self.cache_manager is not None:
            return torch.from_numpy(self.cache_manager.occupancy.mask).to(dtype=torch.bool)
        else:
            if self._occupancy_mask is None:
                self._occupancy_mask = torch.any(self._tensor[PIECE_SLICE] > 0, dim=0)
            return self._occupancy_mask

    def list_occupied(self) -> List[Tuple[Coord, Piece]]:
        """Get list of occupied (coord, piece) pairs - UPDATED: Use iterate_occupied."""
        return list(iterate_occupied(self))

    def enumerate_occupied(self) -> Iterable[Tuple[Coord, Piece]]:
        return iterate_occupied(self)

    def clone(self) -> "Board":
        clone = Board.__new__(Board)
        clone._tensor = self._tensor.clone().detach()
        clone._hash = self._hash
        clone._gen = self._gen
        clone._symmetry_manager = None
        clone._occupancy_mask = None
        clone._occupied_list = None
        clone.cache_manager = None  # Avoid stale cache; reinitialize if needed
        return clone

    def share_memory_(self) -> Board:
        self._tensor.share_memory_()
        return self

    def __repr__(self) -> str:
        return f"Board(tensor={self._tensor.shape}, hash={self.byte_hash()})"

    def apply_move(self, mv: Move) -> bool:
        """Apply move with proper cache synchronization."""
        if self.cache_manager is None:
            raise RuntimeError("Board has no cache_manager – cannot apply_move.")

        from_coord = mv.from_coord
        to_coord   = mv.to_coord

        piece = self.cache_manager.occupancy.get(from_coord)
        if piece is None:
            raise ValueError(f"apply_move: empty from-square {from_coord}")

        # Increment generation BEFORE making changes
        self._gen += 1

        if mv.is_capture:
            self.set_piece(to_coord, None)

        # Handle special moves
        if mv.metadata.get("is_swap", False):
            from game3d.game.move_utils import apply_swap_move
            apply_swap_move(self, mv)
        elif getattr(mv, "is_promotion", False) and getattr(mv, "promotion_type", None):
            from game3d.game.move_utils import apply_promotion_move
            apply_promotion_move(self, mv, piece)
        else:
            # Standard move
            self.set_piece(from_coord, None)
            self.set_piece(to_coord, piece)

        self._hash = None
        self._occupancy_mask = None
        self._occupied_list = None

        # Sync with cache manager
        self.cache_manager.set_piece(from_coord, None)
        self.cache_manager.set_piece(to_coord, piece)

        return True

    def validate_tensor(self) -> bool:
        """Check that every (x,y,z) has at most one 1.0 in piece planes - OPTIMIZED."""
        piece_vals = self._tensor[PIECE_SLICE]
        max_per_square = piece_vals.max(dim=0)[0]
        invalid = (max_per_square > 1.0001).any()  # Handle float precision
        if invalid:
            invalid_positions = torch.nonzero(max_per_square > 1.0001, as_tuple=False).tolist()
            print(f"Invalid positions found: {invalid_positions}")
        return not invalid

    @classmethod
    def startpos(cls) -> "Board":
        b = cls.empty()
        b.init_startpos()
        return b

    def _validate_board_state(self) -> None:
        """Validate that all required attributes are present."""
        required_attrs = ['_tensor', '_hash', '_gen', '_occupancy_mask', '_occupied_list', '_symmetry_manager']
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise AttributeError(f"Board missing required attributes: {missing}")

    @property
    def generation(self) -> int:
        """Get the current generation number of the board."""
        return self._gen

    # def enumerate(self) -> Iterable[Tuple[Coord, Optional[Piece]]]:
    #     """Yield (coord, piece_or_None) for every square on the 9×9×9 board."""
    #     for z in range(SIZE_Z):
    #         for y in range(SIZE_Y):
    #             for x in range(SIZE_X):
    #                 c = (x, y, z)
    #                 yield c, self.piece_at(c)

    def set_piece(self, coord: Coord, piece: Optional[Piece]) -> None:
        """Set piece at coord, updating tensor."""
        z, y, x = coord[2], coord[1], coord[0]
        self._tensor[:, z, y, x] = 0.0  # Clear all planes first
        if piece is not None:
            offset = 0 if piece.color == Color.WHITE else N_PIECE_TYPES
            self._tensor[offset + piece.ptype.value, z, y, x] = 1.0
        self._hash = None
        self._occupancy_mask = None
        self._occupied_list = None
        self._gen += 1

    def piece_at(self, coord: Coord) -> Optional[Piece]:
        """Get piece at coord from tensor."""
        if not in_bounds(coord):
            return None
        z, y, x = coord[2], coord[1], coord[0]
        white_vals = self._tensor[0:N_PIECE_TYPES, z, y, x]
        black_vals = self._tensor[N_PIECE_TYPES:2 * N_PIECE_TYPES, z, y, x]
        white_sum = white_vals.sum().item()
        black_sum = black_vals.sum().item()
        if white_sum + black_sum == 0:
            return None
        if white_sum > 0 and black_sum > 0:
            raise ValueError(f"Overlapping pieces at {coord}")
        if white_sum > 0:
            ptype = PieceType(white_vals.argmax().item())
            return Piece(Color.WHITE, ptype)
        else:
            ptype = PieceType(black_vals.argmax().item())
            return Piece(Color.BLACK, ptype)

    def byte_hash(self) -> int:
        """Compute hash."""
        return hash_board_tensor(self._tensor)

    def tensor(self) -> torch.Tensor:
        """Return the raw (C, 9, 9, 9) tensor."""
        return self._tensor

    def get(self, coord: Coord) -> Optional[Piece]:
        """Fallback for old code that still calls board.get(coord)."""
        return self.piece_at(coord)
