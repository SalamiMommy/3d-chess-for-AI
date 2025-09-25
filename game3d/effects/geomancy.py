"""Geomancy – controller may block unoccupied squares within 3-sphere for 5 plies."""

from __future__ import annotations
from typing import List, Tuple, Dict
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto


def block_candidates(board: BoardProto, controller: Color) -> List[Tuple[int, int, int]]:
    """All unoccupied squares within 3-sphere of any friendly GEOMANCER."""
    out: List[Tuple[int, int, int]] = []
    centres = [
        coord for coord, p in board.list_occupied()
        if p.color == controller and p.ptype == PieceType.GEOMANCER
    ]
    for centre in centres:
        for sq in sphere_centre(board, centre, radius=3):
            if board.piece_at(sq) is None:   # unoccupied only
                out.append(sq)
    return out
