# movepiece.py - OPTIMIZED VERSION
"""Optimized Move class – targeting create_batch hotspot."""

import struct
from typing import Optional, Tuple, List, Any, Dict
import numpy as np
import logging
from game3d.common.coord_utils import _COORD_TO_IDX, filter_valid_coords
from game3d.common.enums import PieceType
import numba as nb

MOVE_FLAGS = {
    'CAPTURE': 1 << 0,
    'PROMOTION': 1 << 1,
    'EN_PASSANT': 1 << 2,
    'CASTLE': 1 << 3,
    'ARCHERY': 1 << 4,
    'HIVE': 1 << 5,
    'SELF_DETONATE': 1 << 6,
    'FROZEN': 1 << 7,
    'BUFFED': 1 << 8,
    'DEBUFFED': 1 << 9,
    'GEOMANCY': 1 << 10,
}

# ------------------------------------------------------------------
# OPTIMIZED MovePool - PRE-ALLOCATES MEMORY
# ------------------------------------------------------------------
class MovePool:
    __slots__ = ('_pool', '_in_use', '_total_created', '_max_in_use', '_preallocated')
    MAX_POOL_SIZE = 2_000_000  # Increased for batch processing
    PREALLOCATE_SIZE = 1_000_000

    def __init__(self):
        self._pool = []
        self._in_use = set()
        self._total_created = 0
        self._max_in_use = 0
        self._preallocated = []  # Pre-allocated array for batch operations
        self._preallocate_large()

    def _preallocate_large(self):
        """Pre-allocate a large block of moves for batch operations"""
        block = [self._new_move() for _ in range(self.PREALLOCATE_SIZE)]
        self._pool.extend(block)
        self._total_created += self.PREALLOCATE_SIZE

    def _new_move(self) -> 'Move':
        m = Move.__new__(Move)
        m._data = 0
        m._cached_hash = None
        m.metadata = {}
        # Pre-initialize coordinate caches
        m._cached_from = (0, 0, 0)
        m._cached_to = (0, 0, 0)
        return m

    def populate(self, n: int) -> None:
        """Pre-create n instances."""
        if len(self._pool) < n:
            additional = n - len(self._pool)
            self._pool.extend([self._new_move() for _ in range(additional)])
            self._total_created += additional

    def acquire(self):
        if self._pool:
            move = self._pool.pop()
        else:
            move = self._new_move()
            self._total_created += 1

        self._in_use.add(id(move))
        self._max_in_use = max(self._max_in_use, len(self._in_use))
        return move

    def release(self, move):
        move_id = id(move)
        if move_id in self._in_use:
            self._in_use.remove(move_id)

            # Fast reset - only clear essential fields
            move.metadata.clear()
            move._cached_hash = None

            if len(self._pool) < self.MAX_POOL_SIZE:
                self._pool.append(move)

    def acquire_batch(self, n: int) -> List['Move']:
        """ULTRA-FAST batch acquisition with pre-allocation."""
        if n <= 0:
            return []

        # Use pre-allocated block if available
        if n <= len(self._pool):
            moves = self._pool[-n:]
            del self._pool[-n:]
        else:
            # Take all available and create the rest
            moves = self._pool[:]
            self._pool.clear()
            needed = n - len(moves)
            moves.extend([self._new_move() for _ in range(needed)])
            self._total_created += needed

        # Batch add to in_use set
        self._in_use.update(id(m) for m in moves)
        self._max_in_use = max(self._max_in_use, len(self._in_use))

        return moves

    def get_stats(self):
        return {
            'total_created': self._total_created,
            'pool_size': len(self._pool),
            'in_use': len(self._in_use),
            'max_in_use': self._max_in_use
        }

_move_pool_instance = None

def _get_move_pool() -> MovePool:
    """Get the singleton MovePool instance."""
    global _move_pool_instance
    if _move_pool_instance is None:
        _move_pool_instance = MovePool()
    return _move_pool_instance

# ------------------------------------------------------------------
# OPTIMIZED Move class with vectorized operations
# ------------------------------------------------------------------
class Move:
    __slots__ = ('_data', '_cached_hash', 'metadata', '_cached_from', '_cached_to')

    @classmethod
    def create_batch(
        cls, from_coord: Tuple[int, int, int],
        to_coords: np.ndarray,
        captures: np.ndarray,
        debuffed: bool = False,
    ) -> List['Move']:
        """ULTRA-OPTIMIZED batch creation with minimal overhead."""
        n = len(to_coords)
        if n == 0:
            return []

        # Ensure to_coords has correct dtype for the calculations
        to_coords = to_coords.astype(np.uint16)  # Add this line

        # Use pre-validated coordinates to avoid duplicate checks
        # Assume coordinates are already validated by caller

        # Vectorized coordinate processing
        from_idx = _COORD_TO_IDX[from_coord]

        # Single-pass coordinate conversion and validation
        x, y, z = to_coords[:, 0], to_coords[:, 1], to_coords[:, 2]
        to_idxs = x + y * 9 + z * 81

        # Vectorized flag computation - ensure flags use appropriate dtype
        flags_base = MOVE_FLAGS['DEBUFFED'] if debuffed else 0
        flags = np.where(captures, flags_base | MOVE_FLAGS['CAPTURE'], flags_base)

        # Use uint32 for the data computation to avoid overflow
        datas = from_idx | (to_idxs.astype(np.uint32) << 10) | (flags.astype(np.uint32) << 20)

        # Batch acquire and initialize
        moves = _get_move_pool().acquire_batch(n)

        # Pre-convert to list for faster access
        to_list = to_coords.tolist()

        # Minimal initialization loop
        for i in range(n):
            move = moves[i]
            move._data = int(datas[i])
            move._cached_hash = None
            move._cached_from = from_coord
            move._cached_to = tuple(to_list[i])
            # Skip metadata initialization - will be done on demand

        return moves
    # ----------------------------------------------------------
    # introspection
    # ----------------------------------------------------------
    @property
    def is_capture(self) -> bool:
        return bool(self._data & (MOVE_FLAGS['CAPTURE'] << 20))

    @property
    def is_promotion(self) -> bool:
        return bool(self._data & (MOVE_FLAGS['PROMOTION'] << 20))

    @property
    def is_buffed(self) -> bool:
        return bool(self._data & (MOVE_FLAGS['BUFFED'] << 20))

    @property
    def is_debuffed(self) -> bool:
        return bool(self._data & (MOVE_FLAGS['DEBUFFED'] << 20))

    @property
    def flags(self) -> int:
        return (self._data >> 20) & 0xFFF

    @property
    def from_coord(self) -> Tuple[int, int, int]:
        # Use pre-cached coordinate if available
        if hasattr(self, '_cached_from'):
            return self._cached_from

        # Fallback to computation only when needed
        idx = self._data & 0x3FF
        z, rem = divmod(idx, 81)
        y, x = divmod(rem, 9)
        coord = (x, y, z)
        self._cached_from = coord
        return coord

    @property
    def to_coord(self) -> Tuple[int, int, int]:
        if hasattr(self, '_cached_to'):
            return self._cached_to

        to_idx = (self._data >> 10) & 0x3FF
        z, rem = divmod(to_idx, 81)
        y, x = divmod(rem, 9)
        coord = (x, y, z)
        self._cached_to = coord
        return coord

    # ----------------------------------------------------------
    # dunder + life-cycle
    # ----------------------------------------------------------
    def __hash__(self) -> int:
        if self._cached_hash is None:
            self._cached_hash = hash(self._data)
        return self._cached_hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Move) and self._data == other._data

    def __repr__(self) -> str:
        fx, fy, fz = self.from_coord
        tx, ty, tz = self.to_coord
        capture = "x" if self.is_capture else "-"
        buffed = " (buffed)" if self.is_buffed else ""
        return f"({fx},{fy},{fz}){capture}({tx},{ty},{tz}){buffed}"

    def release(self) -> None:
        """Return this instance to the pool."""
        _get_move_pool().release(self)

    @property
    def promotion_type(self) -> int:
        """Extract promotion piece type from packed data (6 bits sufficient for PieceType values)."""
        return (self._data >> 38) & 0x3F

    @property
    def captured_piece(self) -> int:
        """Extract captured piece type from packed data."""
        return (self._data >> 32) & 0x3F

# ------------------------------------------------------------------
# MoveReceipt (unchanged)
# ------------------------------------------------------------------
class MoveReceipt:
    __slots__ = (
        'new_state',
        'is_legal',
        'is_game_over',
        'result',
        'message',
        'move_time_ms',
        'cache_stats',
    )

    def __init__(
        self,
        new_state: Any,
        is_legal: bool,
        is_game_over: bool,
        result: Optional[Any],
        message: str = "",
        move_time_ms: float = 0.0,
        cache_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        return self.is_legal


# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------
__all__ = [
    'Move',
    'MoveReceipt',
    'MOVE_FLAGS',
    'Move',
]
