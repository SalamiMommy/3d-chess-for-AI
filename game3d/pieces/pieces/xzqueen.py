"""Master definition for XZ-Queen – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.xzqueenmoves import xzqueen_dispatcher

DISPATCHER = xzqueen_dispatcher
CACHES = []
