# game3d/cache/piececache.py
from typing import Dict, Optional, Tuple
from game3d.board.board import Board, WHITE_SLICE, BLACK_SLICE
from game3d.pieces.piece import Piece
from game3d.pieces.enums import Color, PieceType
from game3d.common.common import Coord

class PieceCache:
    __slots__ = ("_cache", "_valid")  # Keep original slots

    def __init__(self, board: Board) -> None:
        self._cache: Dict[Coord, Optional[Piece]] = {}
        self._valid = False
        self.rebuild(board)

    def get(self, coord: Coord) -> Optional[Piece]:
        # Since OptimizedCacheManager always calls rebuild(board) after mutations,
        # we can assume the cache is always valid when get() is called
        return self._cache.get(coord)

    def rebuild(self, board: Board) -> None:
        """Rebuild cache from board tensor (called by Board)."""
        self._cache.clear()
        white_planes = board._tensor[WHITE_SLICE]
        black_planes = board._tensor[BLACK_SLICE]

        # Vectorized extraction (much faster than per-coordinate calls)
        white_occ = (white_planes == 1.0).nonzero()
        for ptype_idx, z, y, x in white_occ.tolist():
            self._cache[(x, y, z)] = Piece(Color.WHITE, PieceType(ptype_idx))

        black_occ = (black_planes == 1.0).nonzero()
        for ptype_idx, z, y, x in black_occ.tolist():
            self._cache[(x, y, z)] = Piece(Color.BLACK, PieceType(ptype_idx))

        self._valid = True

    def invalidate(self) -> None:
        """Mark cache as stale (called after board mutations)."""
        self._valid = False
