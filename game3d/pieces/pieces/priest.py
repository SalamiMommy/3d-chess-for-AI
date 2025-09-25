"""Master definition for Priest – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.priestmoves import priest_dispatcher

DISPATCHER = priest_dispatcher
CACHES = []  # no auras
