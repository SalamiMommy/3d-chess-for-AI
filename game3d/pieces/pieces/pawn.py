"""Raw Pawn data: step vectors, attack vectors, specials."""

from typing import Final
from pieces.enums import PieceType

TYPE: Final[PieceType] = PieceType.TYPE_00      # assign your enum here

# 3D step vectors (dx, dy, dz)
PUSH_VECTORS: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 1, 0),   # white forward
    # (0,-1, 0), # black forward – handled by colour multiplier
)

CAPTURE_VECTORS: Final[tuple[tuple[int, int, int], ...]] = (
    (1, 1, 0),
    (-1, 1, 0),
)

# starting row mask (z, y) – used for double-step
DOUBLE_STEP_Y: Final[int] = 1          # white example
