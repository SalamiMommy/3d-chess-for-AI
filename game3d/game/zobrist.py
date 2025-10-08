# zobrist.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from functools import lru_cache
import random
import threading

# Import Board only for type checking, not at runtime
if TYPE_CHECKING:
    from game3d.board.board import Board
    from game3d.movement.movepiece import Move
    from game3d.cache.manager import OptimizedCacheManager

from game3d.pieces.enums import Color, PieceType
from game3d.common.common import SIZE_X, SIZE_Y, SIZE_Z

# Global Zobrist tables with thread safety
_PIECE_KEYS: Dict[Tuple[PieceType, Color, Tuple[int, int, int]], int] = {}
_EN_PASSANT_KEYS: Dict[Tuple[int, int, int], int] = {}
_CASTLE_KEYS: Dict[str, int] = {}
_SIDE_KEY: int = 0
_INITIALIZED: bool = False
_ZOBRIST_LOCK: threading.RLock = threading.RLock()

def _init_zobrist(width: int = 9, height: int = 9, depth: int = 9) -> None:
    """Thread-safe Zobrist key initialization."""
    global _INITIALIZED, _PIECE_KEYS, _EN_PASSANT_KEYS, _CASTLE_KEYS, _SIDE_KEY

    with _ZOBRIST_LOCK:
        if _INITIALIZED:
            return

        # Use high-quality random numbers
        rng = random.Random(42)  # Fixed seed for reproducibility

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

@lru_cache(maxsize=1024)
def _compute_zobrist_cached(board_hash: int, turn_value: int) -> int:
    """Internal cached function using hashable keys."""
    # This function should not be called directly
    raise NotImplementedError("Use compute_zobrist instead")

def compute_zobrist(board: "Board", color: Color) -> int:
    """Compute Zobrist hash for the board state."""
    _init_zobrist()  # Ensure initialized

    zkey = 0
    # Use board.list_occupied() instead of cache
    for coord, piece in board.list_occupied():
        zkey ^= _PIECE_KEYS[(piece.ptype, piece.color, coord)]

    if color == Color.BLACK:
        zkey ^= _SIDE_KEY

    return zkey

class ZobristHash:
    """Zobrist hashing implementation for incremental hash updates."""

    def __init__(self):
        _init_zobrist()  # Ensure tables are initialized

    def compute_from_scratch(self, board: "Board", color: Color) -> int:
        """Compute Zobrist hash from scratch for a board state."""
        return compute_zobrist(board, color)

    def update_hash_move(
            self,
            current_hash: int,
            mv: "Move",
            from_piece: "Piece",
            captured_piece: Optional["Piece"] = None,
            *,
            cache: Optional["OptimizedCacheManager"] = None,   # NEW kw-only
            **kwargs,                                           # swallow legacy args
    ) -> int:
        """
        Incrementally update the Zobrist hash for a move.
        If *cache* is supplied we use the occupancy cache instead of the board.
        All other keyword arguments (castling, ep, ply) are ignored – they are
        handled by the caller if still needed.
        """
        new_hash = current_hash

        # 1. Remove piece from source square
        new_hash ^= _PIECE_KEYS[(from_piece.ptype, from_piece.color, mv.from_coord)]

        # 2. Handle capture (remove captured piece first)
        if captured_piece is not None:
            new_hash ^= _PIECE_KEYS[(captured_piece.ptype, captured_piece.color, mv.to_coord)]

        # 3. Add piece to destination (promotion already reflected in from_piece)
        new_hash ^= _PIECE_KEYS[(from_piece.ptype, from_piece.color, mv.to_coord)]

        # 4. Flip side-to-move
        new_hash ^= _SIDE_KEY

        return new_hash

    def update_hash_piece_placement(self, current_hash: int, coord: Tuple[int, int, int],
                                  old_piece: Optional["Piece"], new_piece: Optional["Piece"]) -> int:
        """
        Update hash for piece placement changes (add/remove pieces).

        Args:
            current_hash: Current Zobrist hash
            coord: Coordinate where piece changed
            old_piece: Previous piece at coordinate (None if empty)
            new_piece: New piece at coordinate (None if now empty)

        Returns:
            Updated Zobrist hash
        """
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

# For backward compatibility
class CacheDesyncError(Exception):
    """Exception raised when cache desynchronization is detected."""
    pass
