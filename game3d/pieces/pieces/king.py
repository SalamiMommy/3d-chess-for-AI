"""Master definition for King – imports its dispatcher and effect caches."""

from game3d.pieces.enums import PieceType
from game3d.movement.movepieces.kingmoves import king_dispatcher

DISPATCHER = king_dispatcher
CACHES = []  # no auras
