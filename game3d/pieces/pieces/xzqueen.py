"""Master definition for XZ-Queen – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.xzqueenmoves import xzqueen_dispatcher

DISPATCHER = xzqueen_dispatcher
CACHES = []
