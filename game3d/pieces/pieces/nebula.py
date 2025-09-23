"""Master definition for Nebula – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.nebulamoves import nebula_dispatcher

DISPATCHER = nebula_dispatcher
CACHES = []
