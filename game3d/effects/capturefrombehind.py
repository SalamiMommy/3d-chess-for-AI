"""Armoured – WALL may only be captured from behind (opposite face)."""

from __future__ import annotations
from typing import List, Tuple, Set
from game3d.pieces.enums import Color, PieceType
from game3d.common.protocols import BoardProto
from game3d.common.common import add_coords


def _opposite_half_space(centre: Tuple[int, int, int], wall_sq: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """
    Return the 4 “behind” octants (Chebyshev half-space opposite to wall_sq relative to centre).
    Simplest: sign of each coordinate must be **opposite** to the vector centre→wall.
    """
    cx, cy, cz = centre
    wx, wy, wz = wall_sq
    dx = 0 if wx == cx else (1 if wx > cx else -1)
    dy = 0 if wy == cy else (1 if wy > cy else -1)
    dz = 0 if wz == cz else (1 if wz > cz else -1)
    # opposite signs
    ox = -dx if dx != 0 else 0
    oy = -dy if dy != 0 else 0
    oz = -dz if dz != 0 else 0

    # build 4 octants (combinations of opposite signs)
    half: Set[Tuple[int, int, int]] = set()
    for sx in (ox, 0) if ox != 0 else (0,):
        for sy in (oy, 0) if oy != 0 else (0,):
            for sz in (oz, 0) if oz != 0 else (0,):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                half.add((sx, sy, sz))
    return half


def from_behind_squares(board: BoardProto, controller: Color) -> Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]]:
    """
    Return {wall_square: set_of_squares_behind_it} for every friendly WALL.
    If multiple walls overlap, union is taken per wall.
    """
    out: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
    walls = [
        (coord, p) for coord, p in board.list_occupied()
        if p.color == controller and p.ptype == PieceType.WALL
    ]
    for wall_sq, _ in walls:
        behind = set()
        for dx, dy, dz in _opposite_half_space(wall_sq, wall_sq):  # centre = wall itself
            for step in range(1, 9):  # up to board edge
                sq = add_coords(wall_sq, (dx * step, dy * step, dz * step))
                if not in_bounds(sq):
                    break
                behind.add(sq)
        out[wall_sq] = behind
    return out
