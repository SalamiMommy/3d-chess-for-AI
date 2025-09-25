"""Master definition for Swapper – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.swappermoves import swapper_dispatcher

DISPATCHER = swapper_dispatcher
CACHES = []
