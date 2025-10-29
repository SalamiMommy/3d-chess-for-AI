# zobrist.py - FIXED VERSION with consistent instance usage
from __future__ import annotations
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from functools import lru_cache
from threading import RLock
import random

if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.movement.movepiece import Move
    from game3d.pieces.piece import Piece

from game3d.common.enums import Color, PieceType
from game3d.common.constants import SIZE_X, SIZE_Y, SIZE_Z

_PIECE_KEYS: Dict[Tuple[PieceType, Color, Tuple[int, int, int]], int] = {}
_SIDE_KEY: int = 0
_INITIALIZED: bool = False
_INIT_LOCK = RLock()


def _init_zobrist(width: int = 9, height: int = 9, depth: int = 9) -> None:
    """Thread-safe Zobrist key initialization."""
    global _INITIALIZED, _PIECE_KEYS, _SIDE_KEY

    if _INITIALIZED:
        return

    with _INIT_LOCK:
        if _INITIALIZED:
            return

        rng = random.SystemRandom()

        for ptype in PieceType:
            for color in Color:
                for x in range(width):
                    for y in range(height):
                        for z in range(depth):
                            _PIECE_KEYS[(ptype, color, (x, y, z))] = rng.getrandbits(64)

        _SIDE_KEY = rng.getrandbits(64)
        _INITIALIZED = True


_BOARD_HASH_CACHE: Dict[Tuple[int, Color, int], int] = {}
_CACHE_LOCK = RLock()
_MAX_CACHE_SIZE = 10000


def _evict_old_entries() -> None:
    """Remove 20% of oldest cache entries when limit reached."""
    global _BOARD_HASH_CACHE
    if len(_BOARD_HASH_CACHE) > _MAX_CACHE_SIZE:
        items = list(_BOARD_HASH_CACHE.items())
        keep_count = int(_MAX_CACHE_SIZE * 0.8)
        _BOARD_HASH_CACHE = dict(items[-keep_count:])


def compute_zobrist(board: Optional["Board"], color: Color) -> int:
    _init_zobrist()

    if board is None:
        return _SIDE_KEY if color == Color.BLACK else 0

    board_hash = board.byte_hash()
    generation = getattr(board, 'generation', 0)
    cache_key = (board_hash, color, generation)

    with _CACHE_LOCK:
        if cache_key in _BOARD_HASH_CACHE:
            return _BOARD_HASH_CACHE[cache_key]

        # Compute under lock to prevent duplicate work
        zkey = 0

    # Compute outside lock using board's occupancy cache
    # Use the board's method to iterate occupied positions
    for coord, piece in board.list_occupied():
        zkey ^= _PIECE_KEYS[(piece.ptype, piece.color, coord)]

    if color == Color.BLACK:
        zkey ^= _SIDE_KEY

    with _CACHE_LOCK:
        # Double-check pattern
        if cache_key not in _BOARD_HASH_CACHE:
            _BOARD_HASH_CACHE[cache_key] = zkey
            _evict_old_entries()
        return _BOARD_HASH_CACHE[cache_key]

class ZobristHash:
    """Zobrist hashing with incremental updates and smart caching."""

    __slots__ = ('_hash_cache', '_cache_lock', '_piece_keys', '_side_key')

    def __init__(self):
        _init_zobrist()
        self._hash_cache: Dict[int, int] = {}
        self._cache_lock = RLock()
        # Store references to the global keys for instance access
        self._piece_keys = _PIECE_KEYS
        self._side_key = _SIDE_KEY


    def compute_from_scratch(self, board: "Board", color: Color) -> int:
        """Compute Zobrist hash from scratch using instance methods."""
        return compute_zobrist(board, color)

    def update_hash_move(
        self,
        current_hash: int,
        mv: "Move",
        from_piece: "Piece",
        captured_piece: Optional["Piece"] = None,
        **kwargs,
    ) -> int:
        """Incremental update using instance references."""
        # Use instance references to all keys
        new_hash = current_hash

        # Remove piece from source
        new_hash ^= self._piece_keys[(from_piece.ptype, from_piece.color, mv.from_coord)]

        # Remove captured piece if any
        if captured_piece:
            new_hash ^= self._piece_keys[(captured_piece.ptype, captured_piece.color, mv.to_coord)]

        # Add piece to destination (handle promotion)
        final_ptype = getattr(mv, 'promotion_type', from_piece.ptype)
        new_hash ^= self._piece_keys[(final_ptype, from_piece.color, mv.to_coord)]

        # Toggle side to move
        new_hash ^= self._side_key

        return new_hash

    def update_hash_piece_placement(
        self,
        current_hash: int,
        coord: Tuple[int, int, int],
        old_piece: Optional["Piece"],
        new_piece: Optional["Piece"]
    ) -> int:
        """Incrementally update hash for piece placement changes."""
        new_hash = current_hash

        if old_piece:
            new_hash ^= self._piece_keys[(old_piece.ptype, old_piece.color, coord)]

        if new_piece:
            new_hash ^= self._piece_keys[(new_piece.ptype, new_piece.color, coord)]

        return new_hash

    def flip_side(self, current_hash: int) -> int:
        """Flip the side to move in the hash."""
        return current_hash ^ self._side_key

    def clear_cache(self) -> None:
        """Clear the incremental update cache."""
        with self._cache_lock:
            self._hash_cache.clear()

    # Add getters for external access if needed
    def get_piece_key(self, ptype: PieceType, color: Color, coord: Tuple[int, int, int]) -> int:
        """Get piece key for external use (avoid direct global access)."""
        return self._piece_keys[(ptype, color, coord)]

    def get_side_key(self) -> int:
        """Get side key for external use."""
        return self._side_key


def clear_global_zobrist_cache() -> None:
    """Clear the global Zobrist cache (useful for testing or memory management)."""
    global _BOARD_HASH_CACHE
    with _CACHE_LOCK:
        _BOARD_HASH_CACHE.clear()


class CacheDesyncError(Exception):
    """Exception raised when cache desynchronization is detected."""
    pass
