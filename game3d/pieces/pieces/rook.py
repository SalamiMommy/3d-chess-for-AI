"""Master definition for Rook – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.rookmoves import rook_dispatcher

DISPATCHER = rook_dispatcher
CACHES = []
