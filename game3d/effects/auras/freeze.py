"""Freeze aura – enemy pieces inside 2-sphere cannot move."""

from __future__ import annotations
from typing import List, Tuple, Set
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto


def frozen_squares(board: BoardProto, freezer_colour: Color) -> Set[Tuple[int, int, int]]:
    """Return enemy squares within 2-sphere of any friendly FREEZER."""
    frozen: Set[Tuple[int, int, int]] = set()
    for coord, piece in board.list_occupied():
        if piece.color == freezer_colour and piece.ptype == PieceType.FREEZER:
            for sq in sphere_centre(board, coord, radius=2):
                target = cache.piece_cache.get(sq)
                if target is not None and target.color != freezer_colour:
                    frozen.add(sq)
    return frozen
