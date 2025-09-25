"""Master definition for XY-Queen – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.xyqueenmoves import xyqueen_dispatcher

DISPATCHER = xyqueen_dispatcher
CACHES = []
