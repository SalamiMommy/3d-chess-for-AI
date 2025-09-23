"""Master definition for Swapper – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.swappermoves import swapper_dispatcher

DISPATCHER = swapper_dispatcher
CACHES = []
