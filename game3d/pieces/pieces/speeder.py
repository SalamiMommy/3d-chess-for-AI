"""Master definition for Speeder – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.speedermoves import speeder_dispatcher

DISPATCHER = speeder_dispatcher
CACHES = ["movement_buff"]
