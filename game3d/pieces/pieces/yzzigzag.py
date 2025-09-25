"""Master definition for YZ-Zig-Zag – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.yzzigzagmoves import yzzigzag_dispatcher

DISPATCHER = yzzigzag_dispatcher
CACHES = []
