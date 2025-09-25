"""Master definition for 32-Knight – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.knight32moves import knight32_dispatcher

DISPATCHER = knight32_dispatcher
CACHES = []
