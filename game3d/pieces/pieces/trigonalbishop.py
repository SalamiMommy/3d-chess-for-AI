"""Master definition for Trigonal-Bishop – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.trigonalbishopmoves import trigonalbishop_dispatcher

DISPATCHER = trigonalbishop_dispatcher
CACHES = []
