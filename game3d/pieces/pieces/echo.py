"""Master definition for Echo – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.echomoves import echo_dispatcher

DISPATCHER = echo_dispatcher
CACHES = []
