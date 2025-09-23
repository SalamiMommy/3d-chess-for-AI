"""Master definition for Queen – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.queenmoves import queen_dispatcher

DISPATCHER = queen_dispatcher
CACHES = []
