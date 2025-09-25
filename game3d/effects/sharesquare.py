"""Share-Square – Knights may land on occupied squares (friend or foe)."""

from __future__ import annotations
from typing import List, Tuple, Optional
from game3d.pieces.enums import Color, PieceType
from game3d.common.protocols import BoardProto
from game3d.pieces.piece import Piece
from game3d.pieces.enums import PieceType

def is_knight(p: Optional[Piece]) -> bool:
    return p is not None and p.ptype == PieceType.KNIGHT


def can_share(sq: Tuple[int, int, int], board: BoardProto) -> bool:
    """True if square already contains at least one knight (Share-Square active)."""
    # we rely on the cache – this is just a fast scalar check
    return len(board.multi_piece_at(sq)) > 1  # cache provides list
