"""Black-Hole Suck – at end of controller's turn, pull enemies 1 step closer."""

from __future__ import annotations
from typing import List, Tuple, Dict
from pieces.enums import Color
from game3d.effects.auras.aura import sphere_centre, BoardProto
from game.move import Move
from common import add_coords, in_bounds


def _toward(pos: Tuple[int, int, int], target: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """1 Chebyshev step from pos toward target."""
    x, y, z = pos
    tx, ty, tz = target
    dx = 0 if x == tx else (1 if tx > x else -1)
    dy = 0 if y == ty else (1 if ty > y else -1)
    dz = 0 if z == tz else (1 if tz > z else -1)
    return add_coords(pos, (dx, dy, dz))


def suck_candidates(board: BoardProto, controller: Color) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Return dict {enemy_square: pull_target} for every enemy inside 2-sphere of any friendly BLACK_HOLE.
    Pull target is 1 step toward the **nearest** hole (simplest: first hole found).
    """
    out: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    holes: List[Tuple[int, int, int]] = [
        coord for coord, p in board.list_occupied()
        if p.color == controller and p.ptype == PieceType.BLACK_HOLE
    ]
    if not holes:
        return out

    for coord, piece in board.list_occupied():
        if piece.color == controller:
            continue
        for hole in holes:
            if max(abs(coord[0] - hole[0]), abs(coord[1] - hole[1]), abs(coord[2] - hole[2])) <= 2:
                pull = _toward(coord, hole)
                if in_bounds(pull):
                    out[coord] = pull
                break   # pull toward first hole only
    return out
