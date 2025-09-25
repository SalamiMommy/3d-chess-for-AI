"""Generic spherical aura – radius 2 by default."""

from __future__ import annotations
from typing import List, Tuple, Protocol
from game3d.common.common import in_bounds, add_coords


class BoardProto(Protocol):
    def list_occupied(self): ...


def sphere_centre(board: BoardProto, centre: Tuple[int, int, int], radius: int = 2
                 ) -> List[Tuple[int, int, int]]:
    """Return every square within Chebyshev distance ≤ radius."""
    out: List[Tuple[int, int, int]] = []
    cx, cy, cz = centre
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == dy == dz == 0:
                    continue
                sq = add_coords(centre, (dx, dy, dz))
                if in_bounds(sq):
                    out.append(sq)
    return out
