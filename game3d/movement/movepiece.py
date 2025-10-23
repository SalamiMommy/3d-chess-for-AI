# movepiece.py
"""Optimized Move class – targeting create_batch hotspot."""

import struct
from typing import Optional, Tuple, List, Any, Dict
import numpy as np
from game3d.common.coord_utils import _COORD_TO_IDX, filter_valid_coords
from game3d.common.enums import PieceType

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
    # 'DETONATE' not added as new; using 'SELF_DETONATE' for bomb consistency
}

# ------------------------------------------------------------------
# Move-pool management – built *lazily* so Move already exists
# ------------------------------------------------------------------
class MovePool:
    """Light-weight object pool for Move instances."""
    __slots__ = ('_pool', '_in_use')

    def __init__(self) -> None:
        self._pool: List[Move] = []
        self._in_use: set[int] = set()

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
    def acquire(self) -> 'Move':
        if self._pool:
            move = self._pool.pop()
            self._in_use.add(id(move))
            return move
        # pool exhausted – mint one more
        move = self._new_move()
        self._in_use.add(id(move))
        return move

    def release(self, move: 'Move') -> None:
        move_id = id(move)
        if move_id in self._in_use:
            self._in_use.remove(move_id)
            self._pool.append(move)

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
    ) -> 'Move':
        move = _get_move_pool().acquire()
        from_idx = _COORD_TO_IDX[from_coord]
        to_idx = _COORD_TO_IDX[to_coord]
        flags = MOVE_FLAGS['CAPTURE'] if is_capture else 0
        if debuffed:
            flags |= MOVE_FLAGS['DEBUFFED']
        move._data = from_idx | (to_idx << 10) | (flags << 20)
        move._cached_hash = None
        return move

    @classmethod
    def create_batch(
        cls,
        from_coord: Tuple[int, int, int],
        to_coords: np.ndarray,
        captures: np.ndarray,
        debuffed: bool = False,
    ) -> List['Move']:
        n = len(to_coords)
        if n == 0:
            return []

        if not all(0 <= c < 9 for c in from_coord):
            print(f"Invalid from_coord: {from_coord}")
            return []

        # Filter invalid to_coords
        valid_mask = np.all((to_coords >= 0) & (to_coords < 9), axis=1)
        if not np.all(valid_mask):
            print(f"Invalid to_coords found: {to_coords[~valid_mask]}")

        to_coords = to_coords[valid_mask]
        captures = captures[valid_mask]
        n = len(to_coords)
        if n == 0:
            return []

        moves = _get_move_pool().acquire_batch(n)
        from_idx = _COORD_TO_IDX[from_coord]
        flags_base = MOVE_FLAGS['DEBUFFED'] if debuffed else 0

        for i in range(n):
            to_idx = _COORD_TO_IDX[tuple(to_coords[i])]
            flags = flags_base | (MOVE_FLAGS['CAPTURE'] if captures[i] else 0)
            moves[i]._data = from_idx | (to_idx << 10) | (flags << 20)
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
# legacy adaptor (unchanged)
# ------------------------------------------------------------------
def convert_legacy_move_args(
    from_coord,
    to_coord,
    flags=0,
    captured_piece=None,
    is_promotion=False,
    promotion_type=None,
    is_en_passant=False,
    is_castle=False,
    is_archery=False,
    is_hive=False,
    is_self_detonate=False,
    is_capture=False,
    is_buffed=False,
    is_debuffed=False,
    is_frozen=False,  # ADD THIS MISSING PARAMETER
    **kwargs,
):
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
    if is_buffed:
        flags |= MOVE_FLAGS['BUFFED']
    if is_debuffed:
        flags |= MOVE_FLAGS['DEBUFFED']
    if is_frozen:
        flags |= MOVE_FLAGS['FROZEN']

    captured_int = captured_piece.ptype.value if captured_piece else 0
    promotion_int = promotion_type.value if promotion_type else 0  # FIXED: removed .ptype

    return Move(from_coord, to_coord, flags, captured_int, promotion_int)
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
    'convert_legacy_move_args',
]
