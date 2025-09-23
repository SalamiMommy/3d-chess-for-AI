"""Movement-Buff aura – friendly pieces inside 2-sphere gain +1 step."""

from __future__ import annotations
from typing import List, Tuple, Set
from pieces.enums import Color
from game3d.effects.auras.aura import sphere_centre, BoardProto


def buffed_squares(board: BoardProto, buffer_colour: Color) -> Set[Tuple[int, int, int]]:
    """Return friendly squares within 2-sphere of any friendly MOVEMENT_BUFF piece."""
    buffed: Set[Tuple[int, int, int]] = set()
    for coord, piece in board.list_occupied():
        if piece.color == buffer_colour and piece.ptype == PieceType.MOVEMENT_BUFF:
            for sq in sphere_centre(board, coord, radius=2):
                target = board.piece_at(sq)
                if target is not None and target.color == buffer_colour:
                    buffed.add(sq)
    return buffed
