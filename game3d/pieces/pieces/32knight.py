"""Master definition for 32-Knight – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.knight32moves import knight32_dispatcher

DISPATCHER = knight32_dispatcher
CACHES = []
