# movepiece.py
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
# Move-pool management – built *lazily* so Move already exists
# ------------------------------------------------------------------
class MovePool:
    __slots__ = ('_pool', '_in_use', '_total_created', '_max_in_use')
    MAX_POOL_SIZE = 10000

    def __init__(self):
        self._pool = []
        self._in_use = set()
        self._total_created = 0
        self._max_in_use = 0

    # ----------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------
    def _new_move(self) -> 'Move':
        m = Move.__new__(Move)
        m._data = 0                      # <-- add this
        m._cached_hash = None
        m.metadata = {}
        return m

    def populate(self, n: int) -> None:
        """Pre-create n instances."""
        self._pool = [self._new_move() for _ in range(n)]

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------
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

            # Clear move data before pooling
            move.metadata.clear()
            move._cached_hash = None

            if len(self._pool) < self.MAX_POOL_SIZE:
                self._pool.append(move)

    def get_stats(self):
        return {
            'total_created': self._total_created,
            'pool_size': len(self._pool),
            'in_use': len(self._in_use),
            'max_in_use': self._max_in_use
        }

    def acquire_batch(self, n: int) -> List['Move']:
        """Return n pooled Move objects."""
        need = n
        moves: List[Move] = []

        # take as many as we already have
        avail = min(need, len(self._pool))
        for _ in range(avail):
            m = self._pool.pop()
            self._in_use.add(id(m))
            moves.append(m)
        need -= avail

        # manufacture the rest
        for _ in range(need):
            m = self._new_move()
            self._in_use.add(id(m))
            moves.append(m)

        return moves


# ------------------------------------------------------------------
# lazy singleton – built on first use
# ------------------------------------------------------------------
_move_pool: Optional[MovePool] = None


def _get_move_pool() -> MovePool:
    """Thread-safe enough for import time – creates pool once."""
    global _move_pool
    if _move_pool is None:
        _move_pool = MovePool()
        _move_pool.populate(100_000)
    return _move_pool


# ------------------------------------------------------------------
# Move
# ------------------------------------------------------------------
class Move:
    """Immutable, tightly-packed move descriptor."""
    __slots__ = ('_data', '_cached_hash', 'metadata')

    def __init__(
        self,
        from_coord: Tuple[int, int, int],
        to_coord: Tuple[int, int, int],
        flags: int = 0,
        captured_piece: int = 0,
        promotion_type: int = 0,
    ) -> None:
        from_idx = _COORD_TO_IDX[from_coord]
        to_idx = _COORD_TO_IDX[to_coord]
        self._data = (
            from_idx
            | (to_idx << 10)
            | (flags << 20)
            | (captured_piece << 32)  # Shifted to 32 for 12-bit flags
            | (promotion_type << 38)  # Shifted to 38
        )
        self._cached_hash: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

    @classmethod
    def create_simple(
        cls,
        from_coord: Tuple[int, int, int],
        to_coord: Tuple[int, int, int],
        is_capture: bool = False,
        debuffed: bool = False,
        flags: int = 0,  # ADDED: flags parameter
    ) -> 'Move':
        move = _get_move_pool().acquire()
        from_idx = _COORD_TO_IDX[from_coord]
        to_idx = _COORD_TO_IDX[to_coord]
        base_flags = MOVE_FLAGS['CAPTURE'] if is_capture else 0
        if debuffed:
            base_flags |= MOVE_FLAGS['DEBUFFED']
        base_flags |= flags  # ADDED: incorporate provided flags
        move._data = from_idx | (to_idx << 10) | (base_flags << 20)
        move._cached_hash = None
        return move

    @classmethod
    def create_batch(cls, from_coord: Tuple[int, int, int], to_coords: np.ndarray, captures: np.ndarray, debuffed: bool = False) -> List['Move']:
        n = len(to_coords)
        if n == 0:
            return []

        # Validate and filter (existing)
        valid_mask = np.all((to_coords >= 0) & (to_coords < 9), axis=1)
        to_coords = to_coords[valid_mask]
        captures = captures[valid_mask]
        n = len(to_coords)

        if n == 0:
            return []

        # Vectorized idx computation
        from_idx = from_coord[2] * 81 + from_coord[1] * 9 + from_coord[0]
        to_idxs = to_coords[:, 2] * 81 + to_coords[:, 1] * 9 + to_coords[:, 0]

        flags_base = MOVE_FLAGS['DEBUFFED'] if debuffed else 0
        flags = np.full(n, flags_base, dtype=np.int64)
        flags[captures] |= MOVE_FLAGS['CAPTURE']

        moves = _get_move_pool().acquire_batch(n)

        # Numba-accelerated loop for assignment
        # @nb.jit(nopython=True)
        def assign_data(moves_data, from_idx, to_idxs, flags):  # Note: Can't pass objects to numba, so pass a temp array
            for i in range(n):
                moves_data[i] = from_idx | (to_idxs[i] << 10) | (flags[i] << 20)

        moves_data = np.array([0] * n, dtype=np.int64)  # Temp array
        assign_data(moves_data, from_idx, to_idxs, flags)
        for i in range(n):
            moves[i]._data = moves_data[i]
            moves[i]._cached_hash = None

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
        idx = self._data & 0x3FF
        z = idx // 81
        r = idx % 81
        y = r // 9
        x = r % 9
        return (x, y, z)

    @property
    def to_coord(self) -> Tuple[int, int, int]:
        idx = (self._data >> 10) & 0x3FF
        z = idx // 81
        r = idx % 81
        y = r // 9
        x = r % 9
        return (x, y, z)

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
