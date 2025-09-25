"""Master definition for Pawn – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.pawnmoves import pawn_dispatcher

DISPATCHER = pawn_dispatcher
CACHES = []  # no auras
