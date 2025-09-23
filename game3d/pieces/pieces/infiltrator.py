"""Master definition for Infiltrator – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.infiltratormoves import infiltrator_dispatcher

DISPATCHER = infiltrator_dispatcher
CACHES = []
