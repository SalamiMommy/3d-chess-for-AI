"""Project-wide type aliases to avoid circular imports."""

from typing import Tuple, TypeAlias
from .geometry import Coord

MoveTuple: TypeAlias = Tuple[Coord, Coord]   # (from, to)
PlaneIndex: TypeAlias = int                  # 0 .. N_TOTAL_PLANES-1
