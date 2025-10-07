"""Share-Square – Knights may land on occupied squares (friend or foe)."""

from __future__ import annotations
from typing import List, Tuple, Optional, TYPE_CHECKING
from game3d.pieces.enums import Color, PieceType
from game3d.common.protocols import BoardProto
from game3d.pieces.piece import Piece

if TYPE_CHECKING:
    from game3d.cache.manager import OptimizedCacheManager


def is_knight(p: Optional[Piece]) -> bool:
    return p is not None and p.ptype == PieceType.KNIGHT


def can_share(sq: Tuple[int, int, int], board: BoardProto, cache_manager: Optional[OptimizedCacheManager] = None) -> bool:
    """True if square already contains at least one knight (Share-Square active)."""
    # Use cache manager if available
    if cache_manager:
        pieces = cache_manager.piece_cache.pieces_at(sq)
    else:
        # Fallback to board method if cache manager not available
        pieces = board.multi_piece_at(sq)

    # Check if there are multiple pieces (indicating Share-Square is active)
    return len(pieces) > 1


def has_knight(sq: Tuple[int, int, int], board: BoardProto, cache_manager: Optional[OptimizedCacheManager] = None) -> bool:
    """True if square contains at least one knight."""
    # Use cache manager if available
    if cache_manager:
        pieces = cache_manager.piece_cache.pieces_at(sq)
    else:
        # Fallback to board method if cache manager not available
        pieces = board.multi_piece_at(sq)

    # Check if any of the pieces is a knight
    return any(is_knight(piece) for piece in pieces)


def knight_count(sq: Tuple[int, int, int], board: BoardProto, cache_manager: Optional[OptimizedCacheManager] = None) -> int:
    """Count the number of knights at the given square."""
    # Use cache manager if available
    if cache_manager:
        pieces = cache_manager.piece_cache.pieces_at(sq)
    else:
        # Fallback to board method if cache manager not available
        pieces = board.multi_piece_at(sq)

    # Count knights
    return sum(1 for piece in pieces if is_knight(piece))
