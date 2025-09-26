"""Check detector for 9×9×9: king is attacked AND all priests dead."""
#game3d/attacks/check.py
from __future__ import annotations
from typing import Protocol, Optional, runtime_checkable
from game3d.pieces.enums import PieceType, Color


@runtime_checkable
class BoardProto(Protocol):
    def list_occupied(self): ...


# ------------------------------------------------------------------
# pure helpers – zero concrete imports
# ------------------------------------------------------------------
def _any_priest_alive(board: BoardProto, color: Color) -> bool:
    """Fast scan – returns True on first priest found."""
    for _, piece in board.list_occupied():
        if piece.color == color and piece.ptype == PieceType.PRIEST:
            return True
    return False


def square_attacked_by(
    board: BoardProto,
    current_player: Color,
    square: tuple[int, int, int],
    attacker_color: Color,
    cache=None  # Add cache parameter
) -> bool:
    """Is `square` under attack by any piece of `attacker_color`?"""
    from game3d.movement.pseudo_legal import generate_pseudo_legal_moves
    from game3d.game.gamestate import GameState

    tmp_state = GameState.__new__(GameState)
    tmp_state.board = board
    tmp_state.color = attacker_color
    tmp_state.cache = cache  # Use the provided cache

    for mv in generate_pseudo_legal_moves(tmp_state):
        if mv.to_coord == square:
            return True
    return False


def king_in_check(
    board: BoardProto,
    current_player: Color,
    king_color: Color,
    cache=None  # Add cache parameter
) -> bool:
    """King of `king_color` is in check ⇔ attacked AND zero priests alive."""
    if _any_priest_alive(board, king_color):
        return False

    king_pos: Optional[tuple[int, int, int]] = None
    for coord, piece in board.list_occupied():
        if piece.color == king_color and piece.ptype == PieceType.KING:
            king_pos = coord
            break
    if king_pos is None:
        return False

    return square_attacked_by(board, current_player, king_pos, king_color.opposite(), cache)
