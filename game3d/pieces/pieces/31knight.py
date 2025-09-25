"""Master definition for 31-Knight – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.knight31moves import knight31_dispatcher

DISPATCHER = knight31_dispatcher
CACHES = []
