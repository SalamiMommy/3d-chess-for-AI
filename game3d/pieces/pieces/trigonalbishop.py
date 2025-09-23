"""Master definition for Trigonal-Bishop – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.trigonalbishopmoves import trigonalbishop_dispatcher

DISPATCHER = trigonalbishop_dispatcher
CACHES = []
