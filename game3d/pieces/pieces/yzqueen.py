"""Master definition for YZ-Queen – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.yzqueenmoves import yzqueen_dispatcher

DISPATCHER = yzqueen_dispatcher
CACHES = []
