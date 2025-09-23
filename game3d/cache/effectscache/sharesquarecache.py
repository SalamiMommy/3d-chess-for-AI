"""Multi-occupancy cache for Share-Square (Knights only)."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pieces.enums import Color, PieceType
from game3d.board.board import Board
from pieces.piece import Piece
from game.move import Move


class ShareSquareCache:
    __slots__ = ("_stack", "_board")

    def __init__(self, board: Board) -> None:
        self._board = board
        # square -> list of pieces (bottom first)
        self._stack: Dict[Tuple[int, int, int], List[Piece]] = {}
        self._rebuild()  # seed from initial board

    # ---------- public ----------
    def pieces_at(self, sq: Tuple[int, int, int]) -> List[Piece]:
        """All pieces on square (bottom first).  Normal board still shows last item."""
        return self._stack.get(sq, [])

    def add_knight(self, sq: Tuple[int, int, int], knight: Piece) -> None:
        """Knight lands / shares."""
        if knight.ptype != PieceType.KNIGHT:
            raise ValueError("Only knights may share squares")
        self._stack.setdefault(sq, []).append(knight)

    def remove_knight(self, sq: Tuple[int, int, int], knight: Piece) -> None:
        """Knight leaves or is captured."""
        stack = self._stack.get(sq, [])
        try:
            stack.remove(knight)
        except ValueError:
            pass
        if not stack:
            del self._stack[sq]

    def top_piece(self, sq: Tuple[int, int, int]) -> Optional[Piece]:
        """What normal board.piece_at() should return."""
        stack = self._stack.get(sq)
        return stack[-1] if stack else None

    def apply_move(self, mv: Move, mover: Color) -> None:
        self._board.apply_move(mv)
        self._apply_share_logic(mv, mover)  # handle landing / leaving

    def undo_move(self, mv: Move, mover: Color) -> None:
        self._board.undo_move(mv)
        self._undo_share_logic(mv, mover)   # mirror of apply

    # ---------- internals ----------
    def _rebuild(self) -> None:
        """Seed from initial board – any square with ≥2 knights is multi."""
        self._stack.clear()
        for coord, piece in self._board.list_occupied():
            if piece.ptype == PieceType.KNIGHT:
                self._stack.setdefault(coord, []).append(piece)

    def _apply_share_logic(self, mv: Move, mover: Color) -> None:
        """Mirror the physical move into the multi-stack."""
        from_sq, to_sq = mv.from_coord, mv.to_coord
        mover_piece = self._board.piece_at(from_sq)  # after board.apply_move
        victim_piece = self._board.piece_at(to_sq)   # after board.apply_move

        # 1. mover leaves from_sq
        if mover_piece is not None and mover_piece.ptype == PieceType.KNIGHT:
            self.remove_knight(from_sq, mover_piece)

        # 2. capture landing on a share-square → remove enemy knight(s) first
        if mv.is_capture and victim_piece is not None and victim_piece.ptype == PieceType.KNIGHT:
            self.remove_knight(to_sq, victim_piece)

        # 3. mover lands / shares
        if mover_piece is not None and mover_piece.ptype == PieceType.KNIGHT:
            self.add_knight(to_sq, mover_piece)

    def _undo_share_logic(self, mv: Move, mover: Color) -> None:
        """Exact mirror of _apply_share_logic."""
        # reverse order: un-land, un-capture, un-leave
        from_sq, to_sq = mv.from_coord, mv.to_coord
        mover_piece = self._board.piece_at(to_sq)  # after undo
        victim_piece = self._board.piece_at(from_sq)  # after undo

        if mover_piece is not None and mover_piece.ptype == PieceType.KNIGHT:
            self.remove_knight(to_sq, mover_piece)

        if mv.is_capture and victim_piece is not None and victim_piece.ptype == PieceType.KNIGHT:
            self.add_knight(to_sq, victim_piece)  # restore captured knight

        if mover_piece is not None and mover_piece.ptype == PieceType.KNIGHT:
            self.add_knight(from_sq, mover_piece)  # restore mover


# ------------------------------------------------------------------
# singleton
# ------------------------------------------------------------------
_share_cache: Optional[ShareSquareCache] = None


def init_share_square_cache(board: Board) -> None:
    global _share_cache
    _share_cache = ShareSquareCache(board)


def get_share_square_cache() -> ShareSquareCache:
    if _share_cache is None:
        raise RuntimeError("ShareSquareCache not initialised")
    return _share_cache
