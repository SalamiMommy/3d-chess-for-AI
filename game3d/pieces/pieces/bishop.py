"""Master definition for Bishop – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.bishopmoves import bishop_dispatcher

DISPATCHER = bishop_dispatcher
CACHES = []
