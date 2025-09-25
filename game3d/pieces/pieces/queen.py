"""Master definition for Queen – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.queenmoves import queen_dispatcher

DISPATCHER = queen_dispatcher
CACHES = []
