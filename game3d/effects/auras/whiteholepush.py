"""White-Hole Push – at end of controller's turn, push enemies 1 step away."""

from __future__ import annotations
from typing import List, Tuple, Dict
from game3d.pieces.enums import Color, PieceType
from game3d.effects.auras.aura import sphere_centre, BoardProto
from game3d.movement.movepiece import Move
from game3d.common.common import add_coords, in_bounds


def _away(pos: Tuple[int, int, int], target: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """1 Chebyshev step from pos **away** from target."""
    x, y, z = pos
    tx, ty, tz = target
    dx = 0 if x == tx else (-1 if tx > x else 1)
    dy = 0 if y == ty else (-1 if ty > y else 1)
    dz = 0 if z == tz else (-1 if tz > z else 1)
    return add_coords(pos, (dx, dy, dz))


def push_candidates(board: BoardProto, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Return dict {enemy_square: push_target} for every enemy inside 2-sphere of any friendly WHITE_HOLE.
    Push target is 1 step **away** from the **nearest** hole (first hole found).
    """
    out: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    holes: List[Tuple[int, int, int]] = [
        coord for coord, p in board.list_occupied()
        if p.color == controller and p.ptype == PieceType.WHITEHOLE
    ]
    if not holes:
        return out

    for coord, piece in board.list_occupied():
        if piece.color == controller:
            continue
        for hole in holes:
            if max(abs(coord[0] - hole[0]), abs(coord[1] - hole[1]), abs(coord[2] - hole[2])) <= 2:
                push = _away(coord, hole)
                if in_bounds(push):
                    out[coord] = push
                break   # push away from first hole only
    return out
