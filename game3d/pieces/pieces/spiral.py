"""Master definition for Spiral – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.spiralmoves import spiral_dispatcher

DISPATCHER = spiral_dispatcher
CACHES = []
