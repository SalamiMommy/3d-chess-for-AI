"""Master definition for Wall – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.wallmoves import wall_dispatcher

DISPATCHER = wall_dispatcher
CACHES = ["capture_from_behind"]  # Wall special rule
