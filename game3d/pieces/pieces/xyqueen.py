"""Master definition for XY-Queen – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.xyqueenmoves import xyqueen_dispatcher

DISPATCHER = xyqueen_dispatcher
CACHES = []
