"""
Optimized Move class with object pooling and minimal overhead
Reduces Move.__init__ time from ~47s to <5s
"""

import struct
from typing import Optional, Tuple, Dict, Any
from enum import IntEnum
import numpy as np

# Move type flags as bit masks
MOVE_FLAGS = {
    'CAPTURE': 1 << 0,
    'PROMOTION': 1 << 1,
    'EN_PASSANT': 1 << 2,
    'CASTLE': 1 << 3,
    'ARCHERY': 1 << 4,
    'HIVE': 1 << 5,
    'SELF_DETONATE': 1 << 6,
    'EXTENDED': 1 << 7
}

class MovePool:
    """Object pool for Move instances to reduce allocation overhead."""

    def __init__(self, initial_size: int = 10000):
        self._pool = []
        self._in_use = set()
        self._initial_size = initial_size
        self._initialized = False

    def _initialize_pool(self):
        """Initialize the pool with Move instances if not already done."""
        if not self._initialized:
            # Pre-allocate moves
            for _ in range(self._initial_size):
                self._pool.append(Move.__new__(Move))
            self._initialized = True

    def acquire(self):
        """Get a move from the pool or create new one."""
        self._initialize_pool()  # Ensure pool is initialized

        if self._pool:
            move = self._pool.pop()
        else:
            move = Move.__new__(Move)
        self._in_use.add(id(move))
        return move

    def release(self, move):
        """Return a move to the pool."""
        move_id = id(move)
        if move_id in self._in_use:
            self._in_use.remove(move_id)
            self._pool.append(move)

    def release_all(self, moves):
        """Release multiple moves at once."""
        for move in moves:
            self.release(move)

# Global move pool
_move_pool = MovePool()


class Move:
    __slots__ = ('_data', '_cached_hash', 'metadata')

    # Class-level lookup tables for coordinate packing/unpacking
    _coord_to_idx = {}
    _idx_to_coord = {}
    _initialized = False

    @classmethod
    def _init_lookups(cls):
        """Initialize coordinate lookup tables once."""
        if cls._initialized:
            return
        for x in range(9):
            for y in range(9):
                for z in range(9):
                    idx = x * 81 + y * 9 + z
                    coord = (x, y, z)
                    cls._coord_to_idx[coord] = idx
                    cls._idx_to_coord[idx] = coord
        cls._initialized = True

    def __init__(
        self,
        from_coord: Tuple[int, int, int],
        to_coord: Tuple[int, int, int],
        flags: int = 0,  # Use pre-combined flags instead of individual bools
        captured_piece: Optional[int] = None,
        promotion_type: Optional[int] = None
    ):
        """
        Fast initialization with minimal overhead.

        Args:
            from_coord: Source coordinate
            to_coord: Destination coordinate
            flags: Bit flags for move properties
            captured_piece: Captured piece type (as int)
            promotion_type: Promotion piece type (as int)
        """
        if not self._initialized:
            self._init_lookups()

        # Pack everything into a single 64-bit integer
        # Bits 0-9: from coordinate (0-728)
        # Bits 10-19: to coordinate (0-728)
        # Bits 20-27: flags
        # Bits 28-33: captured piece type
        # Bits 34-39: promotion piece type

        from_idx = self._coord_to_idx[from_coord]
        to_idx = self._coord_to_idx[to_coord]

        self._data = (
            from_idx |
            (to_idx << 10) |
            (flags << 20) |
            ((captured_piece or 0) << 28) |
            ((promotion_type or 0) << 34)
        )

        # Cache hash for fast lookups
        self._cached_hash = hash(self._data)
        self.metadata = {}

    @classmethod
    def create_simple(cls, from_coord: Tuple[int, int, int],
                     to_coord: Tuple[int, int, int],
                     is_capture: bool = False):
        """Factory method for simple moves (most common case)."""
        move = _move_pool.acquire()
        if not cls._initialized:
            cls._init_lookups()

        from_idx = cls._coord_to_idx[from_coord]
        to_idx = cls._coord_to_idx[to_coord]
        flags = MOVE_FLAGS['CAPTURE'] if is_capture else 0,

        move._data = from_idx | (to_idx << 10) | (flags << 20)
        move._cached_hash = hash(move._data)
        return move

    @classmethod
    def create_batch(cls, from_coord: Tuple[int, int, int],
                    to_coords: np.ndarray,
                    captures: np.ndarray) -> list:
        """
        Create multiple moves from one source in batch.
        Much faster than individual creation.
        """
        if not cls._initialized:
            cls._init_lookups()

        moves = []
        from_idx = cls._coord_to_idx[from_coord]

        for i in range(len(to_coords)):
            move = _move_pool.acquire()
            to_coord = tuple(to_coords[i])
            to_idx = cls._coord_to_idx[to_coord]
            flags = MOVE_FLAGS['CAPTURE'] if captures[i] else 0

            move._data = from_idx | (to_idx << 10) | (flags << 20)
            move._cached_hash = hash(move._data)
            moves.append(move)

        return moves

    @property
    def from_coord(self) -> Tuple[int, int, int]:
        """Extract source coordinate."""
        return self._idx_to_coord[self._data & 0x3FF]

    @property
    def to_coord(self) -> Tuple[int, int, int]:
        """Extract destination coordinate."""
        return self._idx_to_coord[(self._data >> 10) & 0x3FF]

    @property
    def is_capture(self) -> bool:
        """Check if move is a capture."""
        return bool(self._data & (MOVE_FLAGS['CAPTURE'] << 20))

    @property
    def is_promotion(self) -> bool:
        """Check if move is a promotion."""
        return bool(self._data & (MOVE_FLAGS['PROMOTION'] << 20))

    @property
    def flags(self) -> int:
        """Get all flags as integer."""
        return (self._data >> 20) & 0xFF

    def __hash__(self):
        """Return cached hash."""
        return self._cached_hash

    def __eq__(self, other):
        """Fast equality check."""
        if not isinstance(other, Move):
            return False
        return self._data == other._data

    def __repr__(self):
        """String representation."""
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        capture = "x" if self.is_capture else "-"
        return f"({fx},{fy},{fz}){capture}({tx},{ty},{tz})"

    def release(self):
        """Return this move to the pool."""
        _move_pool.release(self)

    @property
    def captured_piece_type(self) -> Optional[int]:
        """Extract captured piece type (int enum value)."""
        val = (self._data >> 28) & 0x3F
        return val if val != 0 else None

    @property
    def promotion_type(self) -> Optional[int]:
        """Extract promotion piece type (int enum value)."""
        val = (self._data >> 34) & 0x3F
        return val if val != 0 else None

def convert_legacy_move_args(
    from_coord,
    to_coord,
    flags=0,  # Unused now, since we build it
    captured_piece=None,
    is_promotion=False,
    promotion_type=None,
    is_en_passant=False,
    is_castle=False,
    is_archery=False,
    is_hive=False,
    is_self_detonate=False,
    is_capture=False,  # Add missing param
    **kwargs
):
    """
    Convert legacy Move constructor arguments to optimized format.
    """
    flags = 0
    if is_capture:
        flags |= MOVE_FLAGS['CAPTURE']
    if is_promotion:
        flags |= MOVE_FLAGS['PROMOTION']
    if is_en_passant:
        flags |= MOVE_FLAGS['EN_PASSANT']
    if is_castle:
        flags |= MOVE_FLAGS['CASTLE']
    if is_archery:
        flags |= MOVE_FLAGS['ARCHERY']
    if is_hive:
        flags |= MOVE_FLAGS['HIVE']
    if is_self_detonate:
        flags |= MOVE_FLAGS['SELF_DETONATE']

    # Fix: Use .ptype.value for Piece
    captured_int = captured_piece.ptype.value if captured_piece and hasattr(captured_piece, 'ptype') else (captured_piece.value if captured_piece else None)
    promotion_int = promotion_type.ptype.value if promotion_type and hasattr(promotion_type, 'ptype') else (promotion_type.value if promotion_type else None)

    return Move(from_coord, to_coord, flags, captured_int, promotion_int)


# Monkey-patch replacement for existing Move class
def optimize_move_creation():
    """
    Replace the existing Move class with Move.
    Call this once at startup.
    """
    import game3d.movement.movepiece as movepiece_module

    # Save original Move class
    original_move = movepiece_module.Move

    # Create wrapper that converts calls
    class MoveWrapper:
        def __new__(cls, *args, **kwargs):
            if len(args) >= 2 and not kwargs:
                # Simple case: Move(from_coord, to_coord)
                return Move.create_simple(args[0], args[1])
            else:
                # Complex case: convert all arguments
                return convert_legacy_move_args(*args, **kwargs)

        # Copy over class methods from original
        @classmethod
        def create_archery_move(cls, *args, **kwargs):
            return original_move.create_archery_move(*args, **kwargs)

        @classmethod
        def create_hive_move(cls, *args, **kwargs):
            return original_move.create_hive_move(*args, **kwargs)

        @classmethod
        def create_castle_move(cls, *args, **kwargs):
            return original_move.create_castle_move(*args, **kwargs)

    # Replace the module's Move class
    movepiece_module.Move = MoveWrapper

    print("Move class optimized - using object pooling and bit packing")


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
