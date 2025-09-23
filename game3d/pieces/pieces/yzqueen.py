"""Master definition for YZ-Queen – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.yzqueenmoves import yzqueen_dispatcher

DISPATCHER = yzqueen_dispatcher
CACHES = []
