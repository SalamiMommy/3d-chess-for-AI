"""Master definition for Echo – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.echomoves import echo_dispatcher

DISPATCHER = echo_dispatcher
CACHES = []
