"""Master definition for Freezer – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.freezermoves import freezer_dispatcher

DISPATCHER = freezer_dispatcher
CACHES = ["freeze"]
