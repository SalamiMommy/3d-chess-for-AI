"""Master definition for Freezer – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.freezermoves import freezer_dispatcher

DISPATCHER = freezer_dispatcher
CACHES = ["freeze"]
