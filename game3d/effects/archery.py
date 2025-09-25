"""Archery – controller may attack (capture) any enemy within 2-sphere without moving."""

from __future__ import annotations
from typing import List, Tuple
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto


def archery_targets(board: BoardProto, controller: Color) -> List[Tuple[int, int, int]]:
    """All enemy pieces within 2-sphere of any friendly ARCHER."""
    out: List[Tuple[int, int, int]] = []
    centres = [
        coord for coord, p in board.list_occupied()
        if p.color == controller and p.ptype == PieceType.ARCHER
    ]
    seen: set[Tuple[int, int, int]] = set()
    for centre in centres:
        for sq in sphere_centre(board, centre, radius=2):
            if sq in seen:
                continue
            victim = board.piece_at(sq)
            if victim is not None and victim.color != controller:
                out.append(sq)
                seen.add(sq)
    return out
