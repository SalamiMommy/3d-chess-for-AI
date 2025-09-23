"""Master definition for Priest – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.priestmoves import priest_dispatcher

DISPATCHER = priest_dispatcher
CACHES = []  # no auras
