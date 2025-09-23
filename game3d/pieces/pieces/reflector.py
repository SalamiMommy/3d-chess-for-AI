"""Master definition for Reflector – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.reflectormoves import reflector_dispatcher

DISPATCHER = reflector_dispatcher
CACHES = []
