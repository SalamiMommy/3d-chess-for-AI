"""Master definition for Orbiter – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.orbitermoves import orbiter_dispatcher

DISPATCHER = orbiter_dispatcher
CACHES = []
