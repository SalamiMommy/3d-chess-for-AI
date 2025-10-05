from __future__ import annotations
"""
game3d/board/board.py
9×9×9 board – tensor-first, zero-copy, training-ready.
"""
import torch
import numpy as np
from typing import Optional, Tuple, Iterable, List
from game3d.common.common import (
    SIZE_X, SIZE_Y, SIZE_Z, SIZE, VOLUME,
    N_PIECE_TYPES, N_COLOR_PLANES, N_TOTAL_PLANES,
    Coord, in_bounds, coord_to_idx, idx_to_coord,
    hash_board_tensor, PIECE_SLICE, COLOR_SLICE, CURRENT_SLICE, EFFECT_SLICE, N_CHANNELS
)
from game3d.pieces.enums import Color, PieceType
from game3d.pieces.piece import Piece
from game3d.board.symmetry import SymmetryManager
from game3d.movement.movepiece import Move

class Board:
    __slots__ = (
        "_tensor", "_hash", "_symmetry_manager",
        "_occupancy_mask", "_occupied_list", "_gen", "cache_manager"  # Added cache_manager
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
        # FIXED: Don't initialize here to avoid circular imports
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

    # UPDATE all symmetry methods to use the property:
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
        # Clear all piece planes at this coordinate
        self._tensor[PIECE_SLICE, z, y, x] = 0.0
        if p is not None:
            idx = p.ptype.value  # Use .value for IntEnum
            self._tensor[idx, z, y, x] = 1.0
            # Set color in the color mask plane
            if p.color == Color.WHITE:
                self._tensor[N_PIECE_TYPES, z, y, x] = 1.0
            else:
                self._tensor[N_PIECE_TYPES, z, y, x] = 0.0
        self._hash = None
        self._occupancy_mask = None  # Invalidate cache
        self._occupied_list = None   # Invalidate cache

    def piece_at(self, c: Coord) -> Optional[Piece]:
        # Check if cache_manager exists and has piece_cache
        if hasattr(self, 'cache_manager') and self.cache_manager and hasattr(self.cache_manager, 'piece_cache'):
            return self.cache_manager.piece_cache.get(c)

        # Fallback to direct tensor access if cache isn't ready
        x, y, z = c
        piece_planes = self._tensor[PIECE_SLICE, z, y, x]
        max_val, max_idx = torch.max(piece_planes, dim=0)

        if max_val == 1.0:
            ptype = PieceType(max_idx.item())
            color_val = self._tensor[N_PIECE_TYPES, z, y, x]
            color = Color.WHITE if color_val > 0.5 else Color.BLACK
            return Piece(color, ptype)

        return None

    def multi_piece_at(self, sq: Tuple[int, int, int]) -> List[Piece]:
        if hasattr(self, 'cache_manager') and self.cache_manager:
            return self.cache_manager.effects._effect_caches["share_square"].pieces_at(sq)
        else:
            # Fallback for cases without cache manager
            piece = self.piece_at(sq)
            return [piece] if piece else []

    def mirror_z(self) -> Board:
        # Mirror all planes (piece planes, color plane, current player, effect planes)
        mirrored_tensor = self._tensor.flip(dims=(1,))
        return Board(mirrored_tensor)

    def rotate_90(self, k: int = 1) -> Board:
        # Rotate all planes (piece planes, color plane, current player, effect planes)
        rotated_tensor = torch.rot90(self._tensor, k, dims=(2, 3))
        return Board(rotated_tensor)

    def apply_player_plane(self, color: Color) -> None:
        self._tensor[CURRENT_SLICE] = float(color.value)  # Use .value if Color is IntEnum

    def occupancy_mask(self) -> torch.Tensor:
        if self._occupancy_mask is None:
            # Sum all piece planes to get occupancy
            piece_sum = self._tensor[PIECE_SLICE].sum(dim=0)
            self._occupancy_mask = piece_sum > 0
        return self._occupancy_mask

    def list_occupied(self) -> Iterable[Tuple[Coord, Piece]]:
        if self._occupied_list is None:
            occ = self.occupancy_mask()
            indices = torch.nonzero(occ, as_tuple=False)  # (N, 3) with z, y, x
            self._occupied_list = []
            for idx in indices:
                z, y, x = idx.tolist()
                c = (x, y, z)
                p = self.piece_at(c)
                if p is not None:  # Safety check
                    self._occupied_list.append((c, p))
        return iter(self._occupied_list)

    def clone(self) -> Board:
        """
        Create a deep copy of the board with all internal state.

        CRITICAL: This method uses __new__ to bypass __init__, so ALL
        attributes must be explicitly copied or initialized.
        """
        clone = Board.__new__(Board)

        # Core tensor data
        clone._tensor = self._tensor.clone()

        # Hash state (can be reused since tensor is cloned)
        clone._hash = self._hash

        # Generation counter
        clone._gen = getattr(self, '_gen', 0)

        # FIXED: Properly handle symmetry manager
        clone._symmetry_manager = None  # Force lazy init on first use

        # Cache invalidation (force rebuild on first access)
        clone._occupancy_mask = None
        clone._occupied_list = None

        # Do not copy cache_manager
        if hasattr(clone, 'cache_manager'):
            del clone.cache_manager

        return clone

    def share_memory_(self) -> Board:
        self._tensor.share_memory_()
        return self

    def __repr__(self) -> str:
        return f"Board(tensor={self._tensor.shape}, hash={self.byte_hash()})"

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
        self._gen += 1
        return True                     # ← success

    def validate_tensor(self) -> bool:
        """Check that every (x,y,z) has at most one 1.0 in piece planes."""
        valid = True
        invalid_positions = []
        for z in range(SIZE_Z):
            for y in range(SIZE_Y):
                for x in range(SIZE_X):
                    piece_vals = self._tensor[PIECE_SLICE, z, y, x]
                    total_ones = (piece_vals == 1.0).sum().item()
                    if total_ones > 1:
                        invalid_positions.append((x, y, z))
                        valid = False
        if invalid_positions:
            print(f"Invalid positions found: {invalid_positions}")  # Or log; for debugging
        return valid

    @classmethod
    def startpos(cls) -> "Board":
        """Return a board set up for the initial position."""
        b = cls.empty()   # Goes through __init__
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
