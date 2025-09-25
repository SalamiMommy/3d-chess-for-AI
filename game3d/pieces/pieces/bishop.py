"""Master definition for Bishop – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.bishopmoves import bishop_dispatcher

DISPATCHER = bishop_dispatcher
CACHES = []
