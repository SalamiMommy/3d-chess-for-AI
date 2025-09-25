"""Master definition for Spiral – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.spiralmoves import spiral_dispatcher

DISPATCHER = spiral_dispatcher
CACHES = []
