"""Master definition for Pawn – imports its dispatcher and effect caches."""

from pieces.enums import PieceType
from game3d.movement.piecemoves.pawnmoves import pawn_dispatcher

DISPATCHER = pawn_dispatcher
CACHES = []  # no auras
