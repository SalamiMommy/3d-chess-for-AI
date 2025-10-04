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

    def update_hash_move(self, current_hash: int, mv: "Move", from_piece: "Piece",
                        captured_piece: Optional["Piece"] = None,
                        old_castling: int = 0, new_castling: int = 0,
                        old_ep: Optional[Tuple[int, int, int]] = None,
                        new_ep: Optional[Tuple[int, int, int]] = None,
                        old_ply: int = 0, new_ply: int = 0) -> int:
        """
        Incrementally update Zobrist hash for a move.

        Args:
            current_hash: Current Zobrist hash
            mv: The move being applied
            from_piece: Piece that is moving
            captured_piece: Piece being captured (if any)
            old_castling: Old castling rights
            new_castling: New castling rights
            old_ep: Old en passant square
            new_ep: New en passant square
            old_ply: Old ply count
            new_ply: New ply count

        Returns:
            Updated Zobrist hash
        """
        new_hash = current_hash

        # Remove piece from original square
        new_hash ^= _PIECE_KEYS[(from_piece.ptype, from_piece.color, mv.from_coord)]

        # Handle capture
        if captured_piece:
            new_hash ^= _PIECE_KEYS[(captured_piece.ptype, captured_piece.color, mv.to_coord)]

        # Add piece to new square (handle promotion)
        new_ptype = from_piece.ptype
        if hasattr(mv, 'promotion_ptype') and mv.promotion_ptype:
            new_ptype = mv.promotion_ptype

        new_hash ^= _PIECE_KEYS[(new_ptype, from_piece.color, mv.to_coord)]

        # Update side to move
        new_hash ^= _SIDE_KEY

        # Handle castling rights changes
        if old_castling != new_castling:
            new_hash ^= _CASTLE_KEYS[f"{old_castling}"]
            new_hash ^= _CASTLE_KEYS[f"{new_castling}"]

        # Handle en passant changes
        if old_ep != new_ep:
            if old_ep:
                new_hash ^= _EN_PASSANT_KEYS[old_ep]
            if new_ep:
                new_hash ^= _EN_PASSANT_KEYS[new_ep]

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
