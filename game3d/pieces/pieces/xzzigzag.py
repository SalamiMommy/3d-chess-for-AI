"""Master definition for XZ-Zig-Zag – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.xzzigzagmoves import xzzigzag_dispatcher

DISPATCHER = xzzigzag_dispatcher
CACHES = []  # marks slid squares
