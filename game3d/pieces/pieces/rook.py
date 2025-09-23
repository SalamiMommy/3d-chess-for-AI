"""Master definition for Rook – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.rookmoves import rook_dispatcher

DISPATCHER = rook_dispatcher
CACHES = []
