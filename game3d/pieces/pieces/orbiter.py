"""Master definition for Orbiter – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.orbitermoves import orbiter_dispatcher

DISPATCHER = orbiter_dispatcher
CACHES = []
