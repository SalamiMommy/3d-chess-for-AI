"""Master definition for XZ-Zig-Zag – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.xzzigzagmoves import xzzigzag_dispatcher

DISPATCHER = xzzigzag_dispatcher
CACHES = []  # marks slid squares
