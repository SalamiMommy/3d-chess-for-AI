"""Master definition for YZ-Zig-Zag – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.yzzigzagmoves import yzzigzag_dispatcher

DISPATCHER = yzzigzag_dispatcher
CACHES = []
