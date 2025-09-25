"""Master definition for Nebula – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.nebulamoves import nebula_dispatcher

DISPATCHER = nebula_dispatcher
CACHES = []
