"""Master definition for Reflector – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.reflectormoves import reflector_dispatcher

DISPATCHER = reflector_dispatcher
CACHES = []
