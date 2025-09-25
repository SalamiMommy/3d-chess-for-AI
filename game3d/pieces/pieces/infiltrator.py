"""Master definition for Infiltrator – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.infiltratormoves import infiltrator_dispatcher

DISPATCHER = infiltrator_dispatcher
CACHES = []
