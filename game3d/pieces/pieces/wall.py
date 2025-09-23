"""Master definition for Wall – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.wallmoves import wall_dispatcher

DISPATCHER = wall_dispatcher
CACHES = ["capture_from_behind"]  # Wall special rule
