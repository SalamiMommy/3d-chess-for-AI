"""Bomb – detonates on capture OR on self-move to same square; kills enemies in 2-sphere (no kings)."""

from __future__ import annotations
from typing import List, Tuple, Dict
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto
from game3d.movement.movepiece import Move
from game3d.common.common import in_bounds

def detonate(board: BoardProto, trigger_sq: Tuple[int, int, int], current_color: Color) -> List[Tuple[int, int, int]]:
    """
    Return list of squares that were **vacated** (enemy non-king) within 2-sphere of trigger_sq.
    Caller must actually clear those squares on the board.
    """
    cleared: List[Tuple[int, int, int]] = []
    for sq in sphere_centre(board, trigger_sq, radius=2):
        victim = cache.piece_cache.get(sq)
        if victim is None or victim.color == current_color:  # skip friendly / empty
            continue
        if victim.ptype == PieceType.KING:                    # kings immune
            continue
        cleared.append(sq)
    return cleared
