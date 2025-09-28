"""Movement-Debuff aura – enemy pieces inside 2-sphere lose 1 max step (min 1)."""

from __future__ import annotations
from typing import List, Tuple, Set
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto


def debuffed_squares(board: BoardProto, debuffer_colour: Color) -> Set[Tuple[int, int, int]]:
    """Return enemy squares within 2-sphere of any friendly MOVEMENT_DEBUFF piece."""
    debuffed: Set[Tuple[int, int, int]] = set()
    for coord, piece in board.list_occupied():
        if piece.color == debuffer_colour and piece.ptype == PieceType.SLOWER:
            for sq in sphere_centre(board, coord, radius=2):
                target = cache.piece_cache.get(sq)
                if target is not None and target.color != debuffer_colour:
                    debuffed.add(sq)
    return debuffed
