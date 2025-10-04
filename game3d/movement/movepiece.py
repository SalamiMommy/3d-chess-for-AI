"""Optimized Move class with MoveReceipt - fixes missing class."""

from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import struct
import time
from game3d.pieces.enums import PieceType, Color
from game3d.common.common import Coord
from game3d.pieces.piece import Piece

BOARD_SIZE = 9

class MoveType(Enum):
    NORMAL = 0
    CAPTURE = 1
    PROMOTION = 2
    EN_PASSANT = 3
    CASTLE = 4
    ARCHERY = 5
    HIVE = 6
    SPECIAL = 7

class MoveFlags(Enum):
    CAPTURE = 1 << 0
    PROMOTION = 1 << 1
    EN_PASSANT = 1 << 2
    CASTLE = 1 << 3
    ARCHERY = 1 << 4
    HIVE = 1 << 5
    SELF_DETONATE = 1 << 6
    EXTENDED = 1 << 7

# Pre-compute coordinate packing lookup for common coordinates
_COORD_PACK_CACHE = {}
_COORD_UNPACK_CACHE = {}

def _init_coord_caches():
    """Pre-compute packing/unpacking for all valid 9x9x9 coordinates."""
    for x in range(9):
        for y in range(9):
            for z in range(9):
                coord = (x, y, z)
                packed = (x & 0xFF) | ((y & 0xFF) << 8) | ((z & 0xFF) << 16)
                _COORD_PACK_CACHE[coord] = packed
                _COORD_UNPACK_CACHE[packed] = coord

_init_coord_caches()


class Move:
    """High-performance Move class with fixed double-initialization bug."""

    __slots__ = (
        '_from_packed', '_to_packed', '_flags', '_move_type',
        'captured_piece', 'promotion_type', 'move_id', 'metadata',
        'timestamp', 'removed_pieces', 'moved_pieces'
    )

    def __init__(
            self,
            from_coord: Coord,
            to_coord: Coord,
            is_capture: bool = False,
            captured_piece: Optional[PieceType] = None,
            is_promotion: bool = False,
            promotion_type: Optional[PieceType] = None,
            is_en_passant: bool = False,
            is_castle: bool = False,
            is_archery: bool = False,
            is_hive: bool = False,
            is_extended: bool = False,
            move_id: Optional[int] = None,
            removed_pieces: Optional[List[Tuple[Coord, Piece]]] = None,
            moved_pieces: Optional[List[Tuple[Coord, Coord, Piece]]] = None,
            is_self_detonate: bool = False,
            metadata: Optional[Dict[str, Any]] = None,
            timestamp: Optional[float] = None
        ):

        # OPTIMIZED: Use cached packing (removes bounds check overhead)
        self._from_packed = _COORD_PACK_CACHE.get(from_coord)
        self._to_packed = _COORD_PACK_CACHE.get(to_coord)

        if self._from_packed is None or self._to_packed is None:
            # Fallback for invalid coordinates
            raise ValueError(f"Invalid coordinates: {from_coord}, {to_coord}")

        # FIXED: Only call _set_move_properties ONCE
        self._set_move_properties(
            is_capture, is_promotion, is_en_passant, is_castle,
            is_archery, is_hive, is_self_detonate, is_extended
        )

        # Optional properties
        self.captured_piece = captured_piece
        self.promotion_type = promotion_type or (PieceType.QUEEN if is_promotion else None)
        self.move_id = move_id
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()

        # Side effect logs
        self.removed_pieces = removed_pieces or []
        self.moved_pieces = moved_pieces or []

    @property
    def from_coord(self) -> Coord:
        """OPTIMIZED: Use cached unpacking."""
        return _COORD_UNPACK_CACHE[self._from_packed]

    @property
    def to_coord(self) -> Coord:
        """OPTIMIZED: Use cached unpacking."""
        return _COORD_UNPACK_CACHE[self._to_packed]

    @property
    def is_capture(self) -> bool:
        return bool(self._flags & MoveFlags.CAPTURE.value)

    @property
    def is_promotion(self) -> bool:
        return bool(self._flags & MoveFlags.PROMOTION.value)

    @property
    def is_en_passant(self) -> bool:
        return bool(self._flags & MoveFlags.EN_PASSANT.value)

    @property
    def is_castle(self) -> bool:
        return bool(self._flags & MoveFlags.CASTLE.value)

    @property
    def is_archery(self) -> bool:
        return bool(self._flags & MoveFlags.ARCHERY.value)

    @property
    def is_hive(self) -> bool:
        return bool(self._flags & MoveFlags.HIVE.value)

    @property
    def is_self_detonate(self) -> bool:
        return bool(self._flags & MoveFlags.SELF_DETONATE.value)

    @property
    def move_type(self) -> MoveType:
        return self._move_type

    def __eq__(self, other: Any) -> bool:
        """Optimized equality check."""
        if not isinstance(other, Move):
            return False
        return (self._from_packed == other._from_packed and
                self._to_packed == other._to_packed and
                self._flags == other._flags and
                self.promotion_type == other.promotion_type)

    def __hash__(self) -> int:
        """Optimized hash for use in sets and dicts."""
        return hash((
            self._from_packed,
            self._to_packed,
            self._flags,
            self.promotion_type.value if self.promotion_type else 0
        ))

    def __repr__(self) -> str:
        """Compact string representation."""
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        capture = "x" if self.is_capture else "-"
        promo = f"={self.promotion_type.name[0]}" if self.is_promotion and self.promotion_type else ""
        ep = " e.p." if self.is_en_passant else ""
        castle = " O-O" if self.is_castle else ""
        archery = " ARCH" if self.is_archery else ""
        return f"({fx},{fy},{fz}){capture}({tx},{ty},{tz}){promo}{ep}{castle}{archery}"

    def _set_move_properties(
        self,
        is_capture: bool,
        is_promotion: bool,
        is_en_passant: bool,
        is_castle: bool,
        is_archery: bool,
        is_hive: bool,
        is_self_detonate: bool,
        is_extended: bool
    ) -> None:
        """OPTIMIZED: Set move properties using bit flags - compute flags and type together."""
        # Compute flags in one pass
        flags = 0
        if is_capture:
            flags |= MoveFlags.CAPTURE.value
        if is_promotion:
            flags |= MoveFlags.PROMOTION.value
        if is_en_passant:
            flags |= MoveFlags.EN_PASSANT.value
        if is_castle:
            flags |= MoveFlags.CASTLE.value
        if is_archery:
            flags |= MoveFlags.ARCHERY.value
        if is_hive:
            flags |= MoveFlags.HIVE.value
        if is_self_detonate:
            flags |= MoveFlags.SELF_DETONATE.value
        if is_extended:
            flags |= MoveFlags.EXTENDED.value

        self._flags = flags

        # Determine move type with priority order
        if is_archery:
            self._move_type = MoveType.ARCHERY
        elif is_hive:
            self._move_type = MoveType.HIVE
        elif is_castle:
            self._move_type = MoveType.CASTLE
        elif is_en_passant:
            self._move_type = MoveType.EN_PASSANT
        elif is_promotion:
            self._move_type = MoveType.PROMOTION
        elif is_capture:
            self._move_type = MoveType.CAPTURE
        else:
            self._move_type = MoveType.NORMAL

    def to_tuple(self) -> Tuple[int, ...]:
        """Ultra-compact tuple representation for ML/training."""
        return (
            self._from_packed,
            self._to_packed,
            self._flags,
            self.promotion_type.value if self.promotion_type else 0,
            self.captured_piece.value if self.captured_piece else 0,
        )

    @classmethod
    def from_tuple(cls, data: Tuple[int, ...]) -> 'Move':
        """Reconstruct Move from ultra-compact tuple."""
        from_packed, to_packed, flags, promo_value, capture_value = data[:5]

        from_coord = _COORD_UNPACK_CACHE[from_packed]
        to_coord = _COORD_UNPACK_CACHE[to_packed]

        # Extract flags
        is_capture = bool(flags & MoveFlags.CAPTURE.value)
        is_promotion = bool(flags & MoveFlags.PROMOTION.value)
        is_en_passant = bool(flags & MoveFlags.EN_PASSANT.value)
        is_castle = bool(flags & MoveFlags.CASTLE.value)
        is_archery = bool(flags & MoveFlags.ARCHERY.value)
        is_hive = bool(flags & MoveFlags.HIVE.value)
        is_extended = bool(flags & MoveFlags.EXTENDED.value)
        is_self_detonate = bool(flags & MoveFlags.SELF_DETONATE.value)

        promotion_type = PieceType(promo_value) if promo_value != 0 and promo_value in PieceType._value2member_map_ else None
        captured_piece = PieceType(capture_value) if capture_value != 0 and capture_value in PieceType._value2member_map_ else None

        return cls(
            from_coord=from_coord,
            to_coord=to_coord,
            is_capture=is_capture,
            captured_piece=captured_piece,
            is_promotion=is_promotion,
            promotion_type=promotion_type,
            is_en_passant=is_en_passant,
            is_castle=is_castle,
            is_archery=is_archery,
            is_hive=is_hive,
            is_extended=is_extended,
            is_self_detonate=is_self_detonate
        )

    @classmethod
    def _create_special_move(
        cls,
        from_coord: Coord,
        to_coord: Coord,
        move_type: MoveType,
        is_capture: bool = False,
        captured_piece: Optional[PieceType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Move':
        """Internal helper to create special moves with less duplication."""
        kwargs = {
            'from_coord': from_coord,
            'to_coord': to_coord,
            'is_capture': is_capture,
            'captured_piece': captured_piece,
            'metadata': metadata or {}
        }

        if move_type == MoveType.ARCHERY:
            kwargs['is_archery'] = True
            kwargs['metadata']['attack_type'] = 'sphere_surface'
        elif move_type == MoveType.HIVE:
            kwargs['is_hive'] = True
            kwargs['metadata']['batch_move'] = True
        elif move_type == MoveType.CASTLE:
            kwargs['is_castle'] = True
            # Ensure castle_side is in metadata
            if 'castle_side' not in kwargs['metadata']:
                raise ValueError("castle_side must be specified in metadata for castling moves")
            kwargs['metadata']['is_king_side'] = kwargs['metadata']['castle_side'] == 'kingside'

        return cls(**kwargs)

    @classmethod
    def create_archery_move(
        cls,
        archer_coord: Coord,
        target_coord: Coord,
        captured_piece: Optional[PieceType] = None
    ) -> 'Move':
        """Create archery attack move."""
        return cls._create_special_move(
            archer_coord, target_coord, MoveType.ARCHERY,
            is_capture=True, captured_piece=captured_piece
        )

    @classmethod
    def create_hive_move(
        cls,
        from_coord: Coord,
        to_coord: Coord,
        is_capture: bool = False,
        captured_piece: Optional[PieceType] = None
    ) -> 'Move':
        """Create hive piece move."""
        return cls._create_special_move(
            from_coord, to_coord, MoveType.HIVE,
            is_capture=is_capture, captured_piece=captured_piece
        )

    @classmethod
    def create_castle_move(
        cls,
        king_from: Coord,
        king_to: Coord,
        castle_side: str
    ) -> 'Move':
        """Create castling move."""
        return cls._create_special_move(
            king_from, king_to, MoveType.CASTLE,
            metadata={'castle_side': castle_side}
        )
# ==============================================================================
# MOVE RECEIPT - Result object for move submission
# ==============================================================================

class MoveReceipt:
    """Receipt returned after move submission with validation results."""

    __slots__ = (
        'new_state', 'is_legal', 'is_game_over', 'result',
        'message', 'move_time_ms', 'cache_stats'
    )

    def __init__(
        self,
        new_state: Any,  # GameState type hint causes circular import
        is_legal: bool,
        is_game_over: bool,
        result: Optional[Any],  # Result enum
        message: str = "",
        move_time_ms: float = 0.0,
        cache_stats: Optional[Dict[str, Any]] = None
    ):
        self.new_state = new_state
        self.is_legal = is_legal
        self.is_game_over = is_game_over
        self.result = result
        self.message = message
        self.move_time_ms = move_time_ms
        self.cache_stats = cache_stats or {}

    def __repr__(self) -> str:
        status = "LEGAL" if self.is_legal else "ILLEGAL"
        return f"MoveReceipt({status}, {self.move_time_ms:.2f}ms, {self.message})"

    def __bool__(self) -> bool:
        """Allow boolean checks: if receipt: ..."""
        return self.is_legal
