"""Multi-occupancy cache for Share-Square (Knights only)."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from game3d.pieces.enums import Color, PieceType
from game3d.board.board import Board
from game3d.pieces.piece import Piece
from game3d.movement.movepiece import Move

class ShareSquareCache:
    __slots__ = ("_stack",)

    def __init__(self) -> None:
        self._stack: Dict[Tuple[int, int, int], List[Piece]] = {}

    # ---------- public ----------
    def pieces_at(self, sq: Tuple[int, int, int]) -> List[Piece]:
        return self._stack.get(sq, [])

    def add_knight(self, sq: Tuple[int, int, int], knight: Piece) -> None:
        if knight.ptype != PieceType.KNIGHT:
            raise ValueError("Only knights may share squares")
        self._stack.setdefault(sq, []).append(knight)

    def remove_knight(self, sq: Tuple[int, int, int], knight: Piece) -> None:
        stack = self._stack.get(sq, [])
        try:
            stack.remove(knight)
        except ValueError:
            pass
        if not stack:
            del self._stack[sq]

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional[Piece]:
        stack = self._stack.get(sq)
        return stack[-1] if stack else None

    def apply_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._apply_share_logic(mv, mover, board)

    def undo_move(self, mv: Move, mover: Color, board: Board) -> None:
        self._undo_share_logic(mv, mover, board)

    # ---------- internals ----------
    def _rebuild(self, board: Board) -> None:
        self._stack.clear()
        for coord, piece in board.list_occupied():
            if piece.ptype == PieceType.KNIGHT:
                self._stack.setdefault(coord, []).append(piece)

    def _apply_share_logic(self, mv: Move, mover: Color, board: Board) -> None:
        from_sq, to_sq = mv.from_coord, mv.to_coord
        mover_piece = board.piece_at(from_sq)
        victim_piece = board.piece_at(to_sq)
        if mover_piece is not None and mover_piece.ptype == PieceType.KNIGHT:
            self.remove_knight(from_sq, mover_piece)
        if mv.is_capture and victim_piece is not None and victim_piece.ptype == PieceType.KNIGHT:
            self.remove_knight(to_sq, victim_piece)
        if mover_piece is not None and mover_piece.ptype == PieceType.KNIGHT:
            self.add_knight(to_sq, mover_piece)

    def _undo_share_logic(self, mv: Move, mover: Color, board: Board) -> None:
        from_sq, to_sq = mv.from_coord, mv.to_coord
        mover_piece = board.piece_at(to_sq)
        victim_piece = board.piece_at(from_sq)
        if mover_piece is not None and mover_piece.ptype == PieceType.KNIGHT:
            self.remove_knight(to_sq, mover_piece)
        if mv.is_capture and victim_piece is not None and victim_piece.ptype == PieceType.KNIGHT:
            self.add_knight(to_sq, victim_piece)
        if mover_piece is not None and mover_piece.ptype == PieceType.KNIGHT:
            self.add_knight(from_sq, mover_piece)


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_share_cache: Optional[ShareSquareCache] = None

def init_share_square_cache() -> None:
    global _share_cache
    _share_cache = ShareSquareCache()

def get_share_square_cache() -> ShareSquareCache:
    if _share_cache is None:
        raise RuntimeError("ShareSquareCache not initialised")
    return _share_cache
