# zobrist.py - FIXED VERSION with aggressive caching
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
from game3d.common.piece_utils import iterate_occupied

# Global Zobrist tables with thread safety
_PIECE_KEYS: Dict[Tuple[PieceType, Color, Tuple[int, int, int]], int] = {}
_EN_PASSANT_KEYS: Dict[Tuple[int, int, int], int] = {}
_CASTLE_KEYS: Dict[str, int] = {}
_SIDE_KEY: int = 0
_INITIALIZED: bool = False
_INIT_LOCK = RLock()


def _init_zobrist(width: int = 9, height: int = 9, depth: int = 9) -> None:
    """Thread-safe Zobrist key initialization."""
    global _INITIALIZED, _PIECE_KEYS, _EN_PASSANT_KEYS, _CASTLE_KEYS, _SIDE_KEY

    if _INITIALIZED:
        return

    with _INIT_LOCK:
        # Double-check after acquiring lock
        if _INITIALIZED:
            return

        rng = random.SystemRandom()

        # Initialize piece keys
        for ptype in PieceType:
            for color in Color:
                for x in range(width):
                    for y in range(height):
                        for z in range(depth):
                            _PIECE_KEYS[(ptype, color, (x, y, z))] = rng.getrandbits(64)

        # Initialize en passant keys
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    _EN_PASSANT_KEYS[(x, y, z)] = rng.getrandbits(64)

        # Initialize castling keys
        for cr in range(16):
            _CASTLE_KEYS[f"{cr}"] = rng.getrandbits(64)

        _SIDE_KEY = rng.getrandbits(64)
        _INITIALIZED = True


# Cache board hashes with generation tracking
_BOARD_HASH_CACHE: Dict[Tuple[int, Color, int], int] = {}
_CACHE_LOCK = RLock()
_MAX_CACHE_SIZE = 10000


def _evict_old_entries() -> None:
    """Remove 20% of oldest cache entries when limit reached."""
    global _BOARD_HASH_CACHE
    if len(_BOARD_HASH_CACHE) > _MAX_CACHE_SIZE:
        # Simple FIFO eviction (could be LRU with OrderedDict)
        items = list(_BOARD_HASH_CACHE.items())
        keep_count = int(_MAX_CACHE_SIZE * 0.8)
        _BOARD_HASH_CACHE = dict(items[-keep_count:])


def compute_zobrist(board: Optional["Board"], color: Color) -> int:
    """
    Compute Zobrist hash with aggressive caching.
    Cache key: (board_hash, color, generation)
    """
    _init_zobrist()

    if board is None:
        return _SIDE_KEY if color == Color.BLACK else 0

    # Create cache key
    board_hash = board.byte_hash()
    generation = getattr(board, 'generation', 0)
    cache_key = (board_hash, color, generation)

    # Check cache (thread-safe)
    with _CACHE_LOCK:
        if cache_key in _BOARD_HASH_CACHE:
            return _BOARD_HASH_CACHE[cache_key]

    # Cache miss - compute from scratch
    zkey = 0
    for coord, piece in iterate_occupied(board):
        zkey ^= _PIECE_KEYS[(piece.ptype, piece.color, coord)]

    if color == Color.BLACK:
        zkey ^= _SIDE_KEY

    # Store in cache (thread-safe)
    with _CACHE_LOCK:
        _BOARD_HASH_CACHE[cache_key] = zkey
        _evict_old_entries()

    return zkey


class ZobristHash:
    """Zobrist hashing with incremental updates and smart caching."""

    __slots__ = ('_hash_cache', '_cache_lock')

    def __init__(self):
        _init_zobrist()
        # Per-instance cache for incremental updates
        self._hash_cache: Dict[int, int] = {}
        self._cache_lock = RLock()

    def compute_from_scratch(self, board: "Board", color: Color) -> int:
        """Compute Zobrist hash from scratch (uses global cache)."""
        return compute_zobrist(board, color)

    def update_hash_move(
        self,
        current_hash: int,
        mv: "Move",
        from_piece: "Piece",
        captured_piece: Optional["Piece"] = None,
        **kwargs,
    ) -> int:
        """
        Incrementally update Zobrist hash for a move.
        This is the main optimization - avoids full recomputation.

        Performance: O(1) vs O(N) for full recomputation
        """
        # Check if we've cached this exact transformation
        transform_key = hash((current_hash, mv.from_coord, mv.to_coord,
                             from_piece.ptype, from_piece.color,
                             captured_piece.ptype if captured_piece else None,
                             captured_piece.color if captured_piece else None))

        with self._cache_lock:
            if transform_key in self._hash_cache:
                return self._hash_cache[transform_key]

        new_hash = current_hash

        # 1. Remove piece from source square
        new_hash ^= _PIECE_KEYS[(from_piece.ptype, from_piece.color, mv.from_coord)]

        # 2. Handle capture (remove captured piece)
        if captured_piece is not None:
            new_hash ^= _PIECE_KEYS[(captured_piece.ptype, captured_piece.color, mv.to_coord)]

        # 3. Add piece to destination (handle promotion)
        final_ptype = mv.promotion_type if mv.is_promotion else from_piece.ptype
        new_hash ^= _PIECE_KEYS[(final_ptype, from_piece.color, mv.to_coord)]

        # 4. Flip side-to-move
        new_hash ^= _SIDE_KEY

        # Cache the transformation
        with self._cache_lock:
            if len(self._hash_cache) > 5000:
                # Simple eviction - remove half
                items = list(self._hash_cache.items())
                self._hash_cache = dict(items[-2500:])
            self._hash_cache[transform_key] = new_hash

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

        # Remove old piece
        if old_piece:
            new_hash ^= _PIECE_KEYS[(old_piece.ptype, old_piece.color, coord)]

        # Add new piece
        if new_piece:
            new_hash ^= _PIECE_KEYS[(new_piece.ptype, new_piece.color, coord)]

        return new_hash

    def flip_side(self, current_hash: int) -> int:
        """Flip the side to move in the hash."""
        return current_hash ^ _SIDE_KEY

    def clear_cache(self) -> None:
        """Clear the incremental update cache."""
        with self._cache_lock:
            self._hash_cache.clear()


def clear_global_zobrist_cache() -> None:
    """Clear the global Zobrist cache (useful for testing or memory management)."""
    global _BOARD_HASH_CACHE
    with _CACHE_LOCK:
        _BOARD_HASH_CACHE.clear()


class CacheDesyncError(Exception):
    """Exception raised when cache desynchronization is detected."""
    pass
