"""Master definition for Speeder – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.speedermoves import speeder_dispatcher

DISPATCHER = speeder_dispatcher
CACHES = ["movement_buff"]
