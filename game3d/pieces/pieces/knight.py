"""Raw Knight data."""

from typing import Final
from pieces.enums import PieceType

TYPE: Final[PieceType] = PieceType.TYPE_01

LEAP_VECTORS: Final[tuple[tuple[int, int, int], ...]] = (
    ( 2, 1, 0), ( 2,-1, 0),
    (-2, 1, 0), (-2,-1, 0),
    ( 1, 2, 0), ( 1,-2, 0),
    (-1, 2, 0), (-1,-2, 0),
    # 3D extensions – add z ±2, ±1 combinations
    ( 1, 0, 2), ( 1, 0,-2),
    (-1, 0, 2), (-1, 0,-2),
    ( 0, 1, 2), ( 0, 1,-2),
    ( 0,-1, 2), ( 0,-1,-2),
)
