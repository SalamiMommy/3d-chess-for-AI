from __future__ import annotations
"""Optimized Move class for 9×9×9 3D chess with enhanced performance and features."""


from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import struct
import time
import math  # Added import
import re  # Moved to top
from math import gcd
from game3d.pieces.enums import PieceType, Color
from game3d.common.common import Coord
from game3d.pieces.piece import Piece
from game3d.cache.transposition import CompactMove
# ==============================================================================
# OPTIMIZATION CONSTANTS
# ==============================================================================

BOARD_SIZE = 9  # Extracted

class MoveType(Enum):
    """Enumeration of move types for optimized classification."""
    NORMAL = 0
    CAPTURE = 1
    PROMOTION = 2
    EN_PASSANT = 3
    CASTLE = 4
    ARCHERY = 5
    HIVE = 6
    SPECIAL = 7

class MoveFlags(Enum):
    """Bit flags for move properties."""
    CAPTURE = 1 << 0
    PROMOTION = 1 << 1
    EN_PASSANT = 1 << 2
    CASTLE = 1 << 3
    ARCHERY = 1 << 4
    HIVE = 1 << 5
    SELF_DETONATE = 1 << 6
    EXTENDED = 1 << 7

# ==============================================================================
# OPTIMIZED MOVE CLASS
# ==============================================================================

class Move:
    """High-performance Move class with enhanced caching and serialization."""

    __slots__ = (
        # Core coordinates (packed for memory efficiency)
        '_from_packed',      # Packed 3D coordinate
        '_to_packed',        # Packed 3D coordinate

        # Move properties (bit-packed)
        '_flags',            # Bit flags for move properties
        '_move_type',        # MoveType enum value

        # Optional metadata (only when needed)
        'captured_piece',    # Optional[PieceType]
        'promotion_type',    # Optional[PieceType]
        'move_id',          # Optional[int]
        'metadata',         # Optional[Dict[str, Any]]
        'timestamp',        # float

        # Side effect logs (for undo operations)
        'removed_pieces',   # List[Tuple[Coord, Piece]]
        'moved_pieces',     # List[Tuple[Coord, Coord, Piece]]
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
        move_id: Optional[int] = None,
        removed_pieces: Optional[List[Tuple[Coord, Piece]]] = None,
        moved_pieces: Optional[List[Tuple[Coord, Coord, Piece]]] = None,
        is_self_detonate: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        # Validate coords
        for coord in [from_coord, to_coord]:
            if not all(0 <= c < BOARD_SIZE for c in coord):
                raise ValueError(f"Invalid coordinate: {coord}")

        # Pack coordinates for memory efficiency
        self._from_packed = self._pack_coord(from_coord)
        self._to_packed = self._pack_coord(to_coord)

        # Set move type and flags
        self._set_move_properties(
            is_capture, is_promotion, is_en_passant, is_castle,
            False, False, is_self_detonate, False
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

    # ---------- PROPERTIES ----------
    @property
    def from_coord(self) -> Coord:
        return self._unpack_coord(self._from_packed)

    @property
    def to_coord(self) -> Coord:
        return self._unpack_coord(self._to_packed)

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

    # ---------- CORE METHODS ----------
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

    # ---------- SERIALIZATION ----------
    def to_tuple(self) -> Tuple[int, ...]:
        """
        Ultra-compact tuple representation for ML/training.
        Format: (from_packed, to_packed, flags, promo_type, capture_type)
        """
        return (
            self._from_packed,
            self._to_packed,
            self._flags,
            self.promotion_type.value if self.promotion_type else 0,
            self.captured_piece.value if self.captured_piece else 0,
        )

    @classmethod
    def from_tuple(cls, data: Tuple[int, ...]) -> 'Move':
        """
        Reconstruct Move from ultra-compact tuple.
        """
        from_packed, to_packed, flags, promo_value, capture_value = data[:5]

        from_coord = cls._unpack_coord(from_packed)
        to_coord = cls._unpack_coord(to_packed)

        # Extract flags
        is_capture = bool(flags & MoveFlags.CAPTURE.value)
        is_promotion = bool(flags & MoveFlags.PROMOTION.value)
        is_en_passant = bool(flags & MoveFlags.EN_PASSANT.value)
        is_castle = bool(flags & MoveFlags.CASTLE.value)
        is_self_detonate = bool(flags & MoveFlags.SELF_DETONATE.value)

        promotion_type = PieceType(promo_value) if promo_value != 0 and promo_value in PieceType._value2member_map_ else None  # Validate enum
        captured_piece = PieceType(capture_value) if capture_value != 0 and capture_value in PieceType._value2member_map_ else None  # Validate enum

        return cls(
            from_coord=from_coord,
            to_coord=to_coord,
            is_capture=is_capture,
            captured_piece=captured_piece,
            is_promotion=is_promotion,
            promotion_type=promotion_type,
            is_en_passant=is_en_passant,
            is_castle=is_castle,
        )

    def to_bytes(self) -> bytes:
        """Ultra-compact binary representation."""
        return struct.pack(
            '!IIHHH',
            self._from_packed,
            self._to_packed,
            self._flags,
            self.promotion_type.value if self.promotion_type else 0,
            self.captured_piece.value if self.captured_piece else 0,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Move':
        """Reconstruct from binary representation."""
        from_packed, to_packed, flags, promo_value, capture_value = struct.unpack('!IIHHH', data)

        # Reconstruct using from_tuple for consistency
        return cls.from_tuple((from_packed, to_packed, flags, promo_value, capture_value))

    # ---------- UTILITY METHODS ----------
    def manhattan_distance(self) -> int:
        """Fast Manhattan distance calculation."""
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        return abs(tx - fx) + abs(ty - fy) + abs(tz - fz)

    def euclidean_distance(self) -> float:
        """Euclidean distance calculation."""
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        dx, dy, dz = tx - fx, ty - fy, tz - fz
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def direction_vector(self) -> Tuple[int, int, int]:
        """Get normalized direction vector."""
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        dx, dy, dz = tx - fx, ty - fy, tz - fz

        # Normalize to unit steps
        length = math.gcd(math.gcd(dx, dy), dz)  # Use gcd for integer normalization
        if length == 0:
            return (0, 0, 0)

        return (dx // length, dy // length, dz // length)

    def is_forward_pawn_push(self, color: Color) -> bool:
        """Check if move is forward pawn push."""
        if not (self.from_coord[1] + (1 if color == Color.WHITE else -1) == self.to_coord[1]):
            return False
        return (self.from_coord[0] == self.to_coord[0] and
                self.from_coord[2] == self.to_coord[2])

    def is_diagonal_move(self) -> bool:
        """Check if move is diagonal (changes multiple coordinates)."""
        changes = sum(1 for i in range(3) if self.from_coord[i] != self.to_coord[i])
        return changes > 1

    def is_axis_aligned(self) -> bool:
        """Check if move is axis-aligned (changes only one coordinate)."""
        changes = sum(1 for i in range(3) if self.from_coord[i] != self.to_coord[i])
        return changes == 1

    # ---------- SPECIAL MOVE CREATORS ----------
    @classmethod
    def create_archery_move(
        cls,
        archer_coord: Coord,
        target_coord: Coord,
        captured_piece: Optional[PieceType] = None
    ) -> 'Move':
        """Create archery attack move."""
        return cls(
            from_coord=archer_coord,
            to_coord=target_coord,
            is_capture=True,
            captured_piece=captured_piece,
            metadata={'is_archery': True, 'attack_type': 'sphere_surface'}
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
        return cls(
            from_coord=from_coord,
            to_coord=to_coord,
            is_capture=is_capture,
            captured_piece=captured_piece,
            metadata={'is_hive': True, 'batch_move': True}
        )

    @classmethod
    def create_castle_move(
        cls,
        king_from: Coord,
        king_to: Coord,  # Updated to king_to
        castle_side: str
    ) -> 'Move':
        """Create castling move."""
        return cls(
            from_coord=king_from,
            to_coord=king_to,
            is_castle=True,
            metadata={'castle_side': castle_side, 'is_king_side': castle_side == 'kingside'}
        )

    # ---------- INTERNAL METHODS ----------
    @staticmethod
    def _pack_coord(coord: Coord) -> int:
        """Pack 3D coordinate into single integer."""
        x, y, z = coord
        if not all(0 <= c < 256 for c in (x, y, z)):
            raise ValueError(f"Coordinate out of range: {coord}")
        return (x & 0xFF) | ((y & 0xFF) << 8) | ((z & 0xFF) << 16)

    @staticmethod
    def _unpack_coord(packed: int) -> Coord:
        """Unpack single integer into 3D coordinate."""
        x = packed & 0xFF
        y = (packed >> 8) & 0xFF
        z = (packed >> 16) & 0xFF
        return (x, y, z)

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
        """Set move properties using bit flags."""
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

        # Determine move type
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

# ==============================================================================
# MOVE FACTORY FUNCTIONS
# ==============================================================================

def create_move_from_notation(notation: str) -> Move:
    """Create move from standard notation string."""
    # Parse notation like "(1,2,3)-(4,5,6)" or "(1,2,3)x(4,5,6)"
    notation = notation.strip()

    # Extract coordinates
    coord_pattern = r'\((\d+),(\d+),(\d+)\)'
    matches = re.findall(coord_pattern, notation)

    if len(matches) != 2:
        raise ValueError(f"Invalid notation: {notation}")

    from_coord = tuple(int(x) for x in matches[0])
    to_coord = tuple(int(x) for x in matches[1])

    # Determine move properties
    is_capture = 'x' in notation
    is_promotion = '=' in notation
    is_en_passant = 'e.p.' in notation
    is_castle = 'O-O' in notation
    is_archery = 'ARCH' in notation
    is_hive = 'HIVE' in notation

    return Move(
        from_coord=from_coord,
        to_coord=to_coord,
        is_capture=is_capture,
        is_promotion=is_promotion,
        is_en_passant=is_en_passant,
        is_castle=is_castle,
        metadata={
            'from_notation': True,
            'original_notation': notation
        }
    )

# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

def benchmark_move_operations(iterations: int = 1000000) -> Dict[str, float]:
    """Benchmark move operations for performance analysis."""
    import time

    test_move = Move((1, 2, 3), (4, 5, 6), is_capture=True)
    results = {}

    # Benchmark creation
    start = time.perf_counter()
    for _ in range(iterations):
        Move((1, 2, 3), (4, 5, 6))
    results['creation'] = (time.perf_counter() - start) / iterations * 1e9  # nanoseconds

    # Benchmark hashing
    start = time.perf_counter()
    for _ in range(iterations):
        hash(test_move)
    results['hashing'] = (time.perf_counter() - start) / iterations * 1e9

    # Benchmark tuple conversion
    start = time.perf_counter()
    for _ in range(iterations):
        test_move.to_tuple()
    results['tuple_conversion'] = (time.perf_counter() - start) / iterations * 1e9

    return results

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

# Maintain original interface while providing enhanced functionality
def create_move(*args, **kwargs) -> Move:
    """Factory function for backward compatibility."""
    return Move(*args, **kwargs)

# Export original class name for compatibility
__all__ = ['Move', 'CompactMove', 'create_move_from_notation', 'benchmark_move_operations']
